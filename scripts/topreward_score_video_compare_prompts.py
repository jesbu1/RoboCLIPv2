#!/usr/bin/env python3
"""Score one video with TOPReward prompt variants and plot reward curves."""

import argparse
import base64
import csv
import io
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


ENVIRONMENT_TO_INSTRUCTION = {
    "button-press-wall-v2": "Press the button from side",
    "button-press-v2": "Press the button from side",
    "button-press-topdown-v2": "Press the button from top",
    "button-press-topdown-wall-v2": "Press the button from top",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "faucet-close-v2": "Close the faucet",
    "handle-press-side-v2": "Press the handle from side",
    "reach-wall-v2": "Reach the goal",
    "sweep-into-v2": "Sweep the block into the hole",
    "window-close-v2": "Close the window",
}


def infer_instruction(video_path: Path) -> str:
    for part in video_path.parts:
        if part in ENVIRONMENT_TO_INSTRUCTION:
            return ENVIRONMENT_TO_INSTRUCTION[part]
    return video_path.parent.parent.name.replace("-", " ")


def aligned_instruction(instruction: str) -> str:
    stripped = instruction.rstrip()
    if stripped.endswith("."):
        return stripped
    return stripped + "."


def read_video_rgb(path: Path):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to read mp4 files") from exc

    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data() or {}
    fps = float(meta.get("fps") or 20.0)
    frames = []
    try:
        for frame in reader:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            frames.append(np.ascontiguousarray(arr[..., :3].astype(np.uint8)))
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, fps


def center_crop_frame(frame: np.ndarray, crop_size: int = 224) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame = frame[..., :3]
    height, width = frame.shape[:2]
    crop_height = min(crop_size, height)
    crop_width = min(crop_size, width)
    top = max((height - crop_height) // 2, 0)
    left = max((width - crop_width) // 2, 0)
    return np.ascontiguousarray(frame[top : top + crop_height, left : left + crop_width])


def frame_to_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(center_crop_frame(frame)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prefix_batches(frames, max_frames_per_query: int):
    batches = []
    for length in range(1, len(frames) + 1):
        prefix = frames[:length]
        if len(prefix) > max_frames_per_query:
            indices = np.linspace(0, len(prefix) - 1, max_frames_per_query, dtype=int)
            prefix = [prefix[i] for i in indices]
        if len(prefix) == 1:
            prefix = [prefix[0], prefix[0]]
        batches.append([frame_to_b64(frame) for frame in prefix])
    return batches


def health(server_url: str, timeout: float):
    resp = requests.get(server_url.rstrip("/") + "/health", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def post_score(server_url, model_name, frames_b64, instruction, timeout):
    payload = {
        "model": model_name,
        "frames_b64": frames_b64,
        "instruction": instruction,
    }
    resp = requests.post(server_url.rstrip("/") + "/score", json=payload, timeout=timeout)
    resp.raise_for_status()
    return float(resp.json()["logprob"])


def post_score_batch(server_url, model_name, chunk, instruction, timeout):
    payload = {
        "requests": [
            {
                "model": model_name,
                "frames_b64": frames_b64,
                "instruction": instruction,
            }
            for frames_b64 in chunk
        ]
    }
    resp = requests.post(server_url.rstrip("/") + "/score_batch", json=payload, timeout=timeout)
    if resp.status_code == 404:
        return [post_score(server_url, model_name, item, instruction, timeout) for item in chunk]
    resp.raise_for_status()
    return [float(item["logprob"]) for item in resp.json()["scores"]]


def score_curve(server_url, model_name, batches, instruction, timeout, request_batch_size):
    scores = []
    for start in range(0, len(batches), request_batch_size):
        chunk = batches[start : start + request_batch_size]
        scores.extend(post_score_batch(server_url, model_name, chunk, instruction, timeout))
        print(f"  scored {min(start + len(chunk), len(batches))}/{len(batches)}", flush=True)
    return np.asarray(scores, dtype=np.float64)


def forward_diff(scores: np.ndarray, gamma: float, scale: float) -> np.ndarray:
    if len(scores) < 2:
        return np.asarray([], dtype=np.float64)
    return scale * (gamma * scores[1:] - scores[:-1])


def save_csv(path, raw_original, diff_original, raw_aligned, diff_aligned):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prefix_frame",
                "original_raw",
                "original_diff_scaled",
                "aligned_raw",
                "aligned_diff_scaled",
            ]
        )
        n = max(len(raw_original), len(raw_aligned))
        for i in range(n):
            writer.writerow(
                [
                    i + 1,
                    raw_original[i] if i < len(raw_original) else "",
                    diff_original[i - 1] if 0 < i <= len(diff_original) else "",
                    raw_aligned[i] if i < len(raw_aligned) else "",
                    diff_aligned[i - 1] if 0 < i <= len(diff_aligned) else "",
                ]
            )


def draw_frame_strip(ax, frames, title, max_images=8):
    ax.set_title(title)
    ax.axis("off")
    count = min(max_images, len(frames))
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    thumbs = []
    for idx in indices:
        image = Image.fromarray(frames[idx]).resize((160, 120))
        thumbs.append(np.asarray(image))
    strip = np.concatenate(thumbs, axis=1)
    ax.imshow(strip)
    for j, idx in enumerate(indices):
        ax.text(j * 160 + 5, 115, f"f{idx + 1}", color="white", fontsize=8, weight="bold")


def save_plot(path, frames, raw_original, diff_original, raw_aligned, diff_aligned, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_raw_o = np.arange(1, len(raw_original) + 1)
    x_diff_o = np.arange(2, len(raw_original) + 1)
    x_raw_a = np.arange(1, len(raw_aligned) + 1)
    x_diff_a = np.arange(2, len(raw_aligned) + 1)

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    draw_frame_strip(axes[0], frames, "input video frames")

    axes[1].plot(x_raw_o, raw_original, color="#1f77b4", lw=1.8)
    axes[1].set_title("original prompt raw score P[t]")
    axes[1].set_ylabel("logprob")

    axes[2].plot(x_diff_o, diff_original, color="#ff7f0e", lw=1.8)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.45)
    axes[2].set_title("original prompt diff reward scale*(gamma*P[t]-P[t-1])")
    axes[2].set_ylabel("scaled diff")

    axes[3].plot(x_raw_a, raw_aligned, color="#2ca02c", lw=1.8)
    axes[3].set_title("fully aligned prompt raw score P[t]")
    axes[3].set_ylabel("logprob")

    axes[4].plot(x_diff_a, diff_aligned, color="#d62728", lw=1.8)
    axes[4].axhline(0, color="black", lw=0.8, alpha=0.45)
    axes[4].set_title("fully aligned prompt diff reward scale*(gamma*P[t]-P[t-1])")
    axes[4].set_xlabel("prefix frame")
    axes[4].set_ylabel("scaled diff")

    for ax in axes[1:]:
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--original-server-url", required=True)
    parser.add_argument(
        "--aligned-server-url",
        help="Server started with the fully aligned prompt/answer-space variant. "
        "If omitted, the original server URL is reused.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--instruction")
    parser.add_argument("--max-frames-per-query", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument("--diff-gamma", type=float, default=1.0)
    parser.add_argument("--diff-scale", type=float, default=64.5)
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction = args.instruction or infer_instruction(video_path)
    original_instruction = instruction.rstrip()
    aligned_instr = aligned_instruction(instruction)
    aligned_url = args.aligned_server_url or args.original_server_url

    frames, fps = read_video_rgb(video_path)
    batches = prefix_batches(frames, args.max_frames_per_query)

    print(f"video={video_path}", flush=True)
    print(f"frames={len(frames)} fps={fps}", flush=True)
    print(f"original_instruction={original_instruction!r}", flush=True)
    print(f"aligned_instruction={aligned_instr!r}", flush=True)
    print(f"original_server={args.original_server_url}", flush=True)
    print(f"aligned_server={aligned_url}", flush=True)

    original_health = health(args.original_server_url, args.request_timeout)
    aligned_health = health(aligned_url, args.request_timeout)

    print("scoring original prompt curve", flush=True)
    raw_original = score_curve(
        args.original_server_url,
        args.model_name,
        batches,
        original_instruction,
        args.request_timeout,
        args.request_batch_size,
    )
    print("scoring fully aligned prompt curve", flush=True)
    raw_aligned = score_curve(
        aligned_url,
        args.model_name,
        batches,
        aligned_instr,
        args.request_timeout,
        args.request_batch_size,
    )

    diff_original = forward_diff(raw_original, args.diff_gamma, args.diff_scale)
    diff_aligned = forward_diff(raw_aligned, args.diff_gamma, args.diff_scale)

    copied_video = output_dir / video_path.name
    if video_path.resolve() != copied_video.resolve():
        shutil.copy2(video_path, copied_video)

    json_path = output_dir / "topreward_scores_compare.json"
    csv_path = output_dir / "topreward_scores_compare.csv"
    png_path = output_dir / "topreward_scores_compare_5panel.png"

    payload = {
        "created_unix": time.time(),
        "input_video": str(video_path),
        "copied_video": str(copied_video),
        "fps": fps,
        "num_frames": len(frames),
        "model_name": args.model_name,
        "max_frames_per_query": args.max_frames_per_query,
        "diff_gamma": args.diff_gamma,
        "diff_scale": args.diff_scale,
        "original": {
            "server_url": args.original_server_url,
            "server_health": original_health,
            "instruction": original_instruction,
            "raw_scores": raw_original.tolist(),
            "diff_scaled": diff_original.tolist(),
        },
        "aligned": {
            "server_url": aligned_url,
            "server_health": aligned_health,
            "instruction": aligned_instr,
            "raw_scores": raw_aligned.tolist(),
            "diff_scaled": diff_aligned.tolist(),
        },
        "note": (
            "Prompt answer spacing is fixed by each running TOPReward server. "
            "Use separate original/aligned server URLs if you need an exact prompt comparison."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(csv_path, raw_original, diff_original, raw_aligned, diff_aligned)
    save_plot(
        png_path,
        frames,
        raw_original,
        diff_original,
        raw_aligned,
        diff_aligned,
        f"{video_path.parent.parent.name}/{video_path.parent.name}/{video_path.name}",
    )

    print("===== outputs =====", flush=True)
    print(f"OUT_DIR={output_dir}", flush=True)
    print(f"JSON={json_path}", flush=True)
    print(f"CSV={csv_path}", flush=True)
    print(f"PNG={png_path}", flush=True)
    print(f"VIDEO={copied_video}", flush=True)


if __name__ == "__main__":
    main()
