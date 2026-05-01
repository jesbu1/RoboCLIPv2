#!/usr/bin/env python3
"""TOPReward no-chat-template scorer served through vLLM prompt logprobs.

The TOPReward paper scores the log probability of the affirmative answer token
("True") appended to a raw video prompt. This server keeps that formulation:
it does not call the OpenAI chat-completions endpoint and does not apply a chat
template. vLLM is used only as the fast batched inference engine.
"""

import argparse
import base64
import io
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory "
    "that completes the following task: "
)
PROMPT_SUFFIX = " Decide whether the above statement is True or not. The answer is:"
DEFAULT_ANSWER = "True"
DEFAULT_FPS = 2.0


class ScoreRequest(BaseModel):
    frames_b64: List[str]
    instruction: str
    model: Optional[str] = None


class ScoreBatchRequest(BaseModel):
    requests: List[ScoreRequest]


def decode_image(data: str) -> Image.Image:
    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def build_raw_prompt(instruction: str, answer: str, video_placeholder: str) -> str:
    return f"{video_placeholder}{PROMPT_PREFIX}{instruction}{PROMPT_SUFFIX}{answer}"


def answer_token_count(tokenizer, instruction: str, answer: str, video_placeholder: str) -> int:
    prefix = build_raw_prompt(instruction, "", video_placeholder)
    full = build_raw_prompt(instruction, answer, video_placeholder)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)

    if len(full_ids) <= len(prefix_ids):
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    else:
        answer_ids = full_ids[len(prefix_ids) :]

    if not answer_ids:
        raise RuntimeError(f"Answer {answer!r} produced no token ids.")
    return len(answer_ids)


def make_video_data(frames_b64: List[str], fps: float, include_metadata: bool):
    frames = [np.asarray(decode_image(frame), dtype=np.uint8) for frame in frames_b64]
    if not frames:
        raise ValueError("frames_b64 cannot be empty")
    video = np.stack(frames, axis=0)
    if include_metadata:
        return (video, {"fps": fps, "total_num_frames": len(frames)})
    return video


def logprob_value(entry: Any, token_id: int) -> float:
    if entry is None:
        raise KeyError(token_id)

    candidates = []
    if isinstance(entry, dict):
        candidates.extend([entry.get(token_id), entry.get(str(token_id))])
    else:
        try:
            candidates.append(entry[token_id])
        except Exception:
            pass

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "logprob"):
            return float(candidate.logprob)
        if isinstance(candidate, dict) and "logprob" in candidate:
            return float(candidate["logprob"])
        if isinstance(candidate, (float, int)):
            return float(candidate)

    available = list(entry.keys())[:20] if isinstance(entry, dict) else type(entry)
    raise KeyError(f"token_id={token_id} not present in prompt_logprobs; available={available}")


def top_logprobs_payload(entry: Any) -> List[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return []

    payload = []
    for token_id, value in entry.items():
        item = {
            "token_id": int(token_id),
            "logprob": float(getattr(value, "logprob", value.get("logprob") if isinstance(value, dict) else value)),
        }
        decoded = getattr(value, "decoded_token", None)
        if decoded is not None:
            item["token"] = decoded
        rank = getattr(value, "rank", None)
        if rank is not None:
            item["rank"] = rank
        payload.append(item)
    return payload


def score_from_prompt_logprobs(output, answer_count: int) -> Dict[str, Any]:
    prompt_token_ids = list(output.prompt_token_ids or [])
    prompt_logprobs = output.prompt_logprobs
    if not prompt_token_ids or prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_token_ids/prompt_logprobs.")
    if answer_count > len(prompt_token_ids):
        raise RuntimeError(
            f"answer token count {answer_count} exceeds prompt length {len(prompt_token_ids)}."
        )

    answer_token_ids = prompt_token_ids[-answer_count:]
    answer_entries = prompt_logprobs[-answer_count:]
    answer_logprob = 0.0
    for token_id, entry in zip(answer_token_ids, answer_entries):
        answer_logprob += logprob_value(entry, token_id)

    return {
        "logprob": float(answer_logprob),
        "token": DEFAULT_ANSWER,
        "token_ids": [int(token_id) for token_id in answer_token_ids],
        "token_count": int(answer_count),
        "top_logprobs": top_logprobs_payload(answer_entries[-1]),
    }


def make_app(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "limit_mm_per_prompt": {"video": 1},
        "seed": args.seed,
    }
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=args.prompt_logprobs,
    )
    llm_lock = threading.Lock()

    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": "topreward_vllm_prompt_logprobs_no_chat_template",
            "model": args.model,
            "prompt_logprobs": args.prompt_logprobs,
        }

    def score_items(items: List[ScoreRequest]):
        if not items:
            return []

        inputs = []
        answer_counts = []
        for item in items:
            if item.model is not None and item.model != args.model:
                raise HTTPException(
                    status_code=400,
                    detail=f"Server model is {args.model}, request asked for {item.model}",
                )
            if not item.frames_b64:
                raise HTTPException(status_code=400, detail="frames_b64 cannot be empty")

            prompt = build_raw_prompt(item.instruction, args.answer, args.video_placeholder)
            answer_counts.append(
                answer_token_count(tokenizer, item.instruction, args.answer, args.video_placeholder)
            )
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "video": make_video_data(
                            item.frames_b64,
                            args.fps,
                            args.include_video_metadata,
                        )
                    },
                    "multi_modal_uuids": {"video": f"topreward-{uuid.uuid4()}"},
                }
            )

        try:
            with llm_lock:
                outputs = llm.generate(inputs, sampling_params=sampling_params)
            return [
                score_from_prompt_logprobs(output, answer_count)
                for output, answer_count in zip(outputs, answer_counts)
            ]
        except Exception as exc:
            print(f"ERROR /score failed: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/score")
    def score(req: ScoreRequest):
        return score_items([req])[0]

    @app.post("/score_batch")
    def score_batch(req: ScoreBatchRequest):
        return {"scores": score_items(req.requests)}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve no-chat-template TOPReward scores.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-logprobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--answer", default=DEFAULT_ANSWER)
    parser.add_argument("--video-placeholder", default=VIDEO_PLACEHOLDER)
    parser.add_argument(
        "--include-video-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Qwen3-VL in vLLM expects video metadata with FPS.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("VLLM_USAGE_STATS", "0")
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
