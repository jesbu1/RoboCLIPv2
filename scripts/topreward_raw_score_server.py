#!/usr/bin/env python3
"""TOPReward raw teacher-forced scoring server.

This avoids the OpenAI chat-completions path so the prompt is not wrapped in a
chat template. It mirrors the official TOPReward Qwen client scoring logic:
append "True" to the prompt, mask the preceding tokens, and return the
log-probability of the final answer token.
"""

import argparse
import base64
import io
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
import transformers
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory "
    "that completes the following task: "
)
PROMPT_SUFFIX = " Decide whether the above statement is True or not. The answer is: "
DEFAULT_ANSWER = "True"
DEFAULT_FPS = 2.0


class ScoreRequest(BaseModel):
    frames_b64: List[str]
    instruction: str
    model: Optional[str] = None


def decode_image(data: str) -> Image.Image:
    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _candidate_attn_implementations(attn_implementation: str):
    requested = (attn_implementation or "auto").lower()
    if requested in ("", "none", "default"):
        return [None]
    if requested != "auto":
        return [attn_implementation]

    candidates = []
    try:
        import flash_attn  # noqa: F401

        candidates.append("flash_attention_2")
    except Exception:
        pass
    candidates.extend(["sdpa", None])
    return candidates


def load_model(model_name: str, dtype: str, device_map: str, attn_implementation: str):
    processor = transformers.AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    base_model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if dtype:
        base_model_kwargs["torch_dtype"] = dtype

    last_error = None
    attn_candidates = _candidate_attn_implementations(attn_implementation)
    for attn_impl in attn_candidates:
        model_kwargs = dict(base_model_kwargs)
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
        print(
            "[TOPReward raw] trying "
            f"attn_implementation={attn_impl or 'default'}",
            flush=True,
        )
        for class_name in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                model = model_cls.from_pretrained(model_name, **model_kwargs)
                model.eval()
                return processor, model
            except TypeError:
                # Older Transformers versions may not accept torch_dtype="auto".
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("torch_dtype", None)
                try:
                    model = model_cls.from_pretrained(model_name, **fallback_kwargs)
                    model.eval()
                    return processor, model
                except Exception as exc:  # pragma: no cover - depends on cluster env
                    print(
                        "[TOPReward raw] loader failed "
                        f"class={class_name} "
                        f"attn_implementation={attn_impl or 'default'} "
                        f"without torch_dtype: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    last_error = exc
            except Exception as exc:  # pragma: no cover - depends on cluster env
                print(
                    "[TOPReward raw] loader failed "
                    f"class={class_name} "
                    f"attn_implementation={attn_impl or 'default'}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                last_error = exc

    raise RuntimeError(f"Unable to load {model_name}: {last_error}")


def move_inputs_to_model(inputs, model):
    device = next(model.parameters()).device
    try:
        return inputs.to(device)
    except AttributeError:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }


def build_official_qwen_inputs(
    processor,
    images,
    instruction: str,
    answer: str,
    fps: float,
):
    """Match TOPReward's QwenClient.compute_instruction_reward input path."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - depends on cluster env
        raise RuntimeError(
            "qwen_vl_utils is required for raw TOPReward scoring. Install it in "
            "the TOPReward environment or use SERVER_BACKEND=vllm."
        ) from exc

    content = [
        {"type": "video", "video": images, "fps": fps},
        {"type": "text", "text": PROMPT_PREFIX},
    ]
    user_messages = [{"role": "user", "content": content}]

    prompt_chat = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    eos_token = getattr(processor.tokenizer, "eos_token", None)
    if eos_token is not None:
        prompt_chat = prompt_chat.split(eos_token)[0]

    instruction_suffix = f"{instruction}{PROMPT_SUFFIX}{answer}"
    full_text = f"{prompt_chat}{instruction_suffix}"
    image_inputs, video_inputs = process_vision_info(user_messages)

    return processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def compute_answer_logprob(inputs, model, tokenizer, top_logprobs: int):
    """Mirror official TOPReward masking: only the final answer token is scored."""
    labels = inputs["input_ids"].clone()
    prompt_length = inputs["input_ids"].shape[1] - 1
    labels[:, :prompt_length] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

    with torch.inference_mode():
        outputs = model(**inputs, labels=labels)

    logits = outputs.logits[:, :-1, :]
    target_labels = labels[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    mask = target_labels != -100
    safe_targets = target_labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs[mask]
    if masked_log_probs.numel() == 0:
        raise RuntimeError("No answer token was available for TOPReward scoring.")

    positions = mask.nonzero(as_tuple=False)
    batch_idx, token_pos = positions[-1]
    token_id = int(target_labels[batch_idx, token_pos].item())
    answer_logprob = float(masked_log_probs.sum().item())

    answer_distribution = log_probs[batch_idx, token_pos]
    top_k = min(top_logprobs, answer_distribution.numel())
    top_values, top_indices = torch.topk(answer_distribution, k=top_k)
    top_logprobs_payload = [
        {
            "token": tokenizer.decode([int(idx.item())]),
            "logprob": float(value.item()),
        }
        for value, idx in zip(top_values, top_indices)
    ]

    return {
        "logprob": answer_logprob,
        "token": tokenizer.decode([token_id]),
        "token_id": token_id,
        "token_count": int(masked_log_probs.numel()),
        "top_logprobs": top_logprobs_payload,
    }


def make_app(args):
    processor, model = load_model(
        args.model,
        args.dtype,
        args.device_map,
        args.attn_implementation,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "topreward_raw_score"}

    @app.post("/score")
    def score(req: ScoreRequest):
        if req.model is not None and req.model != args.model:
            raise HTTPException(
                status_code=400,
                detail=f"Server model is {args.model}, request asked for {req.model}",
            )
        if not req.frames_b64:
            raise HTTPException(status_code=400, detail="frames_b64 cannot be empty")

        images = [decode_image(frame) for frame in req.frames_b64]
        try:
            inputs = build_official_qwen_inputs(
                processor,
                images,
                req.instruction,
                args.answer,
                args.fps,
            )
            inputs = move_inputs_to_model(inputs, model)
            result = compute_answer_logprob(
                inputs,
                model,
                tokenizer,
                args.top_logprobs,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return result

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve raw TOPReward scores.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("TOPREWARD_ATTN_IMPLEMENTATION", "auto"),
        help=(
            "Attention backend for Transformers loading. The default 'auto' "
            "tries flash_attention_2 only when flash_attn is importable, then "
            "falls back to sdpa/default."
        ),
    )
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--answer", default=DEFAULT_ANSWER)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
