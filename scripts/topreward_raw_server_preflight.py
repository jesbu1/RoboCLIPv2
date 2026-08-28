#!/usr/bin/env python3
"""Lightweight preflight for the raw TOPReward Qwen server.

This intentionally avoids loading the 8B model weights. It checks the pieces
that can be validated on CPU or a small/old GPU: imports, model cache metadata,
Qwen processor prompt formatting, answer-token masking, and whether the requested
attention backend has the needed Python package.
"""

import argparse
import importlib
import os
import py_compile
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List


PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory "
    "that completes the following task: "
)
PROMPT_SUFFIX = " Decide whether the above statement is True or not. The answer is: "


@dataclass
class Check:
    name: str
    ok: bool
    level: str
    detail: str


class Reporter:
    def __init__(self):
        self.checks: List[Check] = []

    def add(self, name: str, ok: bool, level: str, detail: str):
        status = "OK" if ok else level
        print(f"[{status}] {name}: {detail}", flush=True)
        self.checks.append(Check(name=name, ok=ok, level=level, detail=detail))

    def ok(self, name: str, detail: str):
        self.add(name, True, "OK", detail)

    def warn(self, name: str, detail: str):
        self.add(name, False, "WARN", detail)

    def fail(self, name: str, detail: str):
        self.add(name, False, "FAIL", detail)

    def summarize(self) -> int:
        fails = [check for check in self.checks if not check.ok and check.level == "FAIL"]
        warns = [check for check in self.checks if not check.ok and check.level == "WARN"]
        print("========== SUMMARY ==========", flush=True)
        print(f"failures={len(fails)} warnings={len(warns)} total_checks={len(self.checks)}", flush=True)
        if fails:
            for check in fails:
                print(f"FAIL: {check.name}: {check.detail}", flush=True)
            return 1
        if warns:
            print("Preflight passed with warnings. A real server smoke test is still needed.", flush=True)
        else:
            print("Preflight passed. A real server smoke test is still recommended.", flush=True)
        return 0


def section(title: str):
    print(f"\n========== {title} ==========", flush=True)


def import_module(reporter: Reporter, module_name: str, required: bool = True):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        message = f"import failed: {type(exc).__name__}: {exc}"
        if required:
            reporter.fail(f"import {module_name}", message)
        else:
            reporter.warn(f"import {module_name}", message)
        return None
    version = getattr(module, "__version__", "unknown")
    reporter.ok(f"import {module_name}", f"version={version}")
    return module


def run_command(reporter: Reporter, name: str, command: List[str], required: bool = False):
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        if required:
            reporter.fail(name, f"command failed to run: {exc}")
        else:
            reporter.warn(name, f"command failed to run: {exc}")
        return
    output = result.stdout.strip()
    if result.returncode == 0:
        reporter.ok(name, output if output else "command succeeded")
    elif required:
        reporter.fail(name, output if output else f"exit={result.returncode}")
    else:
        reporter.warn(name, output if output else f"exit={result.returncode}")


def check_python_file(reporter: Reporter, path: str):
    if not os.path.exists(path):
        reporter.fail("raw server script exists", f"missing: {path}")
        return
    try:
        py_compile.compile(path, doraise=True)
    except Exception as exc:
        reporter.fail("raw server py_compile", f"{type(exc).__name__}: {exc}")
        return
    reporter.ok("raw server py_compile", path)


def check_torch_cuda(reporter: Reporter, torch_module):
    cuda_version = getattr(torch_module.version, "cuda", None)
    reporter.ok("torch cuda build", f"torch_cuda={cuda_version}")
    if not torch_module.cuda.is_available():
        reporter.warn(
            "GPU availability",
            "torch.cuda.is_available() is false; cannot validate model loading or FA2 runtime here.",
        )
        return

    count = torch_module.cuda.device_count()
    reporter.ok("GPU count", str(count))
    prop = torch_module.cuda.get_device_properties(0)
    capability = prop.major * 10 + prop.minor
    memory_gib = prop.total_memory / (1024**3)
    reporter.ok(
        "GPU[0]",
        f"name={prop.name} capability=sm_{capability} memory={memory_gib:.1f}GiB",
    )

    if capability >= 80:
        reporter.ok(
            "flash_attention_2 GPU architecture",
            "Ampere/Ada/Hopper-class GPU detected; FA2 architecture requirement is plausibly satisfied.",
        )
    else:
        reporter.fail(
            "flash_attention_2 GPU architecture",
            "current GPU is below sm_80. P100/V100 cannot run forced flash_attention_2; use A40/A100/H100.",
        )


def load_processor_and_check_prompt(
    reporter: Reporter,
    transformers,
    model_name: str,
    allow_download: bool,
):
    try:
        from PIL import Image
    except Exception as exc:
        reporter.fail("import PIL.Image", f"{type(exc).__name__}: {exc}")
        return

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        reporter.fail("import qwen_vl_utils.process_vision_info", f"{type(exc).__name__}: {exc}")
        return
    reporter.ok("import qwen_vl_utils.process_vision_info", "available")

    local_files_only = not allow_download
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        hint = "set ALLOW_HF_DOWNLOAD=1 if the checkpoint is not cached and network is allowed"
        reporter.fail(
            "AutoProcessor.from_pretrained",
            f"{type(exc).__name__}: {exc}; local_files_only={local_files_only}; {hint}",
        )
        return
    reporter.ok("AutoProcessor.from_pretrained", f"model={model_name} local_files_only={local_files_only}")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        reporter.fail("processor tokenizer", "processor.tokenizer is missing")
        return
    reporter.ok("processor tokenizer", type(tokenizer).__name__)

    frames = [
        Image.new("RGB", (64, 64), color=(0, 0, 0)),
        Image.new("RGB", (64, 64), color=(32, 32, 32)),
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 2.0},
                {"type": "text", "text": PROMPT_PREFIX},
            ],
        }
    ]
    try:
        prompt_chat = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            prompt_chat = prompt_chat.split(eos_token)[0]
        full_text = (
            f"{prompt_chat}"
            f"dummy task{PROMPT_SUFFIX}True"
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception as exc:
        reporter.fail("official Qwen prompt/tokenization path", f"{type(exc).__name__}: {exc}")
        return

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    labels = input_ids.clone()
    prompt_length = input_ids.shape[1] - 1
    labels[:, :prompt_length] = -100
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    mask = labels[:, 1:] != -100
    token_count = int(mask.sum().item())
    target_labels = labels[:, 1:]
    selected = target_labels[mask]
    if token_count != 1:
        reporter.fail("answer token mask", f"expected one answer token, got token_count={token_count}")
        return
    token_id = int(selected[0].item())
    token_text = tokenizer.decode([token_id])
    reporter.ok(
        "official Qwen prompt/tokenization path",
        f"input_len={input_ids.shape[1]} answer_token_id={token_id} answer_token={token_text!r}",
    )


def check_transformers_qwen(reporter: Reporter, transformers):
    has_qwen = hasattr(transformers, "Qwen3VLForConditionalGeneration")
    if has_qwen:
        reporter.ok("transformers Qwen3VLForConditionalGeneration", "available")
    else:
        reporter.fail(
            "transformers Qwen3VLForConditionalGeneration",
            "missing; raw server is configured to match official TOPReward Qwen loading.",
        )


def main():
    parser = argparse.ArgumentParser(description="Lightweight raw TOPReward server preflight.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--server-script", default="scripts/topreward_raw_score_server.py")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("TOPREWARD_ATTN_IMPLEMENTATION", "auto"),
        help="Attention backend to preflight: auto, sdpa, default, or flash_attention_2.",
    )
    args = parser.parse_args()

    reporter = Reporter()

    section("Runtime")
    print(f"python={sys.executable}", flush=True)
    print(f"python_version={sys.version}", flush=True)
    print(f"cwd={os.getcwd()}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"allow_download={args.allow_download}", flush=True)
    print(f"attn_implementation={args.attn_implementation}", flush=True)
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "CUDA_VISIBLE_DEVICES"):
        print(f"{key}={os.environ.get(key, '')}", flush=True)

    section("Static Files")
    check_python_file(reporter, args.server_script)

    section("Imports")
    torch_module = import_module(reporter, "torch", required=True)
    transformers = import_module(reporter, "transformers", required=True)
    import_module(reporter, "fastapi", required=True)
    import_module(reporter, "uvicorn", required=True)
    import_module(reporter, "PIL", required=True)
    import_module(reporter, "requests", required=True)
    requested_attn = (args.attn_implementation or "auto").lower()
    if requested_attn == "flash_attention_2":
        import_module(reporter, "flash_attn", required=True)
    elif requested_attn == "auto":
        import_module(reporter, "flash_attn", required=False)
    else:
        reporter.ok(
            "flash_attn requirement",
            f"not required for attn_implementation={args.attn_implementation}",
        )

    section("GPU / FlashAttention")
    run_command(reporter, "nvidia-smi", ["nvidia-smi"], required=False)
    if torch_module is not None:
        check_torch_cuda(reporter, torch_module)

    section("Qwen Processor / Tokenization")
    if transformers is not None:
        check_transformers_qwen(reporter, transformers)
        load_processor_and_check_prompt(
            reporter,
            transformers,
            args.model,
            allow_download=args.allow_download,
        )

    return reporter.summarize()


if __name__ == "__main__":
    raise SystemExit(main())
