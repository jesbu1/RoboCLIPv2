#!/usr/bin/env python3
"""ROBOMETER HTTP server that returns both progress and success scores."""

from __future__ import annotations

import argparse
import base64
import io
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


MODEL = None
TOKENIZER = None
BATCH_COLLATOR = None
EXP_CONFIG = None
DEVICE = None
IS_DISCRETE = False
NUM_BINS = 10
INFERENCE_LOCK = threading.Lock()


def flatten_first_sequence(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return flatten_first_sequence(value.tolist())
    if isinstance(value, list):
        if not value:
            return []
        if len(value) == 1 and isinstance(value[0], list):
            return flatten_first_sequence(value[0])
        out: list[float] = []
        for item in value:
            if isinstance(item, list):
                nested = flatten_first_sequence(item)
                if len(nested) == 1:
                    out.append(nested[0])
                else:
                    out.extend(nested)
            else:
                out.append(float(item))
        return out
    return [float(value)]


@asynccontextmanager
async def lifespan(app):
    global MODEL, TOKENIZER, BATCH_COLLATOR, EXP_CONFIG, DEVICE, IS_DISCRETE, NUM_BINS

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RobometerProgressSuccessServer] Loading model on {DEVICE}...", flush=True)
    exp_config, tokenizer, processor, model = load_model_from_hf(
        model_path="aliangdw/Robometer-4B",
        device=DEVICE,
    )
    model.eval()

    MODEL = model
    TOKENIZER = tokenizer
    EXP_CONFIG = exp_config
    BATCH_COLLATOR = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    IS_DISCRETE = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    NUM_BINS = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    print(
        "[RobometerProgressSuccessServer] Model loaded. "
        f"discrete={IS_DISCRETE}, bins={NUM_BINS}",
        flush=True,
    )
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    frames_b64: str
    task: str
    sample_type: str = "progress"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "supports_success": True,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    buf = io.BytesIO(base64.b64decode(req.frames_b64))
    frames = np.load(buf)
    num_frames = frames.shape[0]
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=req.task,
        id="server",
        metadata={"subsequence_length": num_frames},
        video_embeddings=None,
    )
    sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = BATCH_COLLATOR([sample])

    with INFERENCE_LOCK:
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(DEVICE)

        with torch.no_grad():
            results = compute_batch_outputs(
                MODEL,
                TOKENIZER,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=IS_DISCRETE,
                num_bins=NUM_BINS,
            )

        progress = flatten_first_sequence(results.get("progress_pred", []))
        success_outputs = results.get("outputs_success", {}) or {}
        success_probs = flatten_first_sequence(success_outputs.get("success_probs", []))

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "progress": progress or [0.0],
        "success_probs": success_probs,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
