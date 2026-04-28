#!/usr/bin/env python3
"""Per-frame ROBOMETER scoring for real orange-cup videos.

For every video frame t, this script sends the trajectory prefix frames[:t+1]
to a ROBOMETER /predict server. The prefix is uniformly subsampled to
--max-frames frames before sending, but a request is still made for every
original video frame. Outputs include raw scores, forward differences
P[t+1] - P[t], and a three-panel visualization video.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests


DEFAULT_ROOT = Path("/scratch1/haobaizh/real_robot/put_orange_cup_into_the_box")
DEFAULT_TASK = "Put the orange cup in the box"


@dataclass(frozen=True)
class VideoJob:
    split: str
    input_path: Path
    output_dir: Path


def log(message: str) -> None:
    print(message, flush=True)


def read_server_url_from_info(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"server info file is empty: {path}")
    for token in reversed(text.split()):
        if token.startswith("http://") or token.startswith("https://"):
            return token.rstrip("/")
    raise RuntimeError(f"server info file does not contain an http URL: {path}")


def resolve_server_url(args: argparse.Namespace) -> str:
    if args.server_url:
        return args.server_url.rstrip("/")
    if args.info_file is None:
        raise RuntimeError("Either --server-url or --info-file is required.")

    deadline = time.time() + args.info_file_wait_sec
    while True:
        if args.info_file.exists() and args.info_file.stat().st_size > 0:
            return read_server_url_from_info(args.info_file)
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for server info file: {args.info_file}")
        log(f"Waiting for server info file: {args.info_file}")
        time.sleep(5)


def health_check(server_url: str) -> None:
    response = requests.get(f"{server_url}/health", timeout=15)
    response.raise_for_status()
    log(f"server_health={response.text}")


def build_jobs(root: Path, only_split: str) -> list[VideoJob]:
    success_input = root / "sucess"
    if not success_input.exists() and (root / "success").exists():
        success_input = root / "success"
    specs = [
        ("unsuccess", root / "unsuccess", root / "unsuccess_score"),
        ("success", success_input, root / "success_score"),
    ]
    jobs: list[VideoJob] = []
    for split, input_dir, output_dir in specs:
        if only_split != "all" and split != only_split:
            continue
        if not input_dir.exists():
            raise RuntimeError(f"input dir not found: {input_dir}")
        for path in sorted(input_dir.glob("*.mp4")):
            jobs.append(VideoJob(split=split, input_path=path, output_dir=output_dir))
    return jobs


def read_video_rgb(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames decoded from video: {path}")
    return np.stack(frames).astype(np.uint8), fps


def maybe_resize_frames(frames: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return frames
    height, width = frames.shape[1:3]
    longest = max(height, width)
    if longest <= max_edge:
        return frames
    scale = max_edge / longest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = [
        cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.stack(resized).astype(np.uint8)


def prefix_for_step(frames: np.ndarray, step: int, max_frames: int) -> np.ndarray:
    prefix_len = step + 1
    if prefix_len <= max_frames:
        return frames[:prefix_len]
    indices = np.linspace(0, step, max_frames, dtype=int)
    return frames[indices]


def post_progress(
    server_url: str,
    frames: np.ndarray,
    task: str,
    timeout: float,
    retries: int,
) -> float:
    buf = io.BytesIO()
    np.save(buf, frames)
    frames_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    payload = {
        "frames_b64": frames_b64,
        "task": task,
        "sample_type": "progress",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(f"{server_url}/predict", json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            progress = result.get("progress", result.get("reward", 0.0))
            if isinstance(progress, list):
                if not progress:
                    return 0.0
                return float(progress[-1])
            return float(progress)
        except Exception as exc:  # noqa: BLE001 - keep server error for logs.
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = min(10 * attempt, 30)
            log(f"[retry] request failed ({attempt}/{retries}): {exc}; waiting {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError(f"ROBOMETER request failed after {retries} attempts: {last_error}") from last_error


def output_paths(job: VideoJob) -> dict[str, Path]:
    stem = job.input_path.stem
    return {
        "json": job.output_dir / f"{stem}_robometer_scores.json",
        "csv": job.output_dir / f"{stem}_robometer_scores.csv",
        "mp4": job.output_dir / f"{stem}_robometer_score_video.mp4",
        "partial": job.output_dir / f"{stem}_robometer_scores.partial.json",
    }


def load_completed_scores(path: Path, expected_frames: int) -> list[float] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("raw_scores", [])
    if len(scores) == expected_frames:
        return [float(x) for x in scores]
    return None


def load_partial_scores(path: Path) -> list[float]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [float(x) for x in data.get("raw_scores", [])]
    except Exception:
        return []


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_metadata(
    job: VideoJob,
    task: str,
    fps: float,
    total_frames: int,
    max_frames: int,
    request_max_edge: int,
    server_url: str,
    raw_scores: Iterable[float],
) -> dict:
    scores = [float(x) for x in raw_scores]
    forward_diff = np.diff(np.asarray(scores, dtype=np.float64)).tolist()
    return {
        "input_video": str(job.input_path),
        "split": job.split,
        "task": task,
        "server_url": server_url,
        "fps": fps,
        "total_frames": total_frames,
        "max_frames_per_prefix": max_frames,
        "request_max_edge": request_max_edge,
        "raw_scores": scores,
        "forward_diff_pt1_minus_pt": [float(x) for x in forward_diff],
    }


def save_scores_csv(path: Path, raw_scores: list[float]) -> None:
    diffs = np.diff(np.asarray(raw_scores, dtype=np.float64))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "raw_score_p_t", "forward_diff_p_t_plus_1_minus_p_t"])
        for i, score in enumerate(raw_scores):
            diff = "" if i >= len(diffs) else f"{float(diffs[i]):.10f}"
            writer.writerow([i, f"{float(score):.10f}", diff])


def resize_keep_aspect_rgb(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def draw_text(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_curve_panel(
    canvas: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    y_min: float,
    y_max: float,
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 1)
    draw_text(canvas, title, (x0, y0 - 28), scale=0.62, thickness=2)
    draw_text(canvas, ylabel, (x0, y1 + 30), scale=0.5, color=(80, 80, 80))

    if len(values) == 0:
        return

    y_min = float(y_min)
    y_max = float(y_max)
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05

    for frac in (0.0, 0.5, 1.0):
        yy = int(round(y1 - frac * (y1 - y0)))
        cv2.line(canvas, (x0, yy), (x1, yy), (235, 235, 235), 1)
        val = y_min + frac * (y_max - y_min)
        draw_text(canvas, f"{val:.2f}", (x0 - 54, yy + 5), scale=0.42, color=(90, 90, 90))

    if y_min < 0.0 < y_max:
        zero_y = int(round(y1 - ((0.0 - y_min) / (y_max - y_min)) * (y1 - y0)))
        cv2.line(canvas, (x0, zero_y), (x1, zero_y), (160, 160, 160), 1)

    last = min(frame_idx, len(values) - 1)
    denom = max(1, len(values) - 1)
    points = []
    for i in range(last + 1):
        x = int(round(x0 + (i / denom) * (x1 - x0)))
        clipped = min(max(float(values[i]), y_min), y_max)
        y = int(round(y1 - ((clipped - y_min) / (y_max - y_min)) * (y1 - y0)))
        points.append((x, y))
    if len(points) == 1:
        cv2.circle(canvas, points[0], 4, color, -1)
    else:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2)
        cv2.circle(canvas, points[-1], 4, color, -1)


def save_score_video(
    path: Path,
    frames_rgb: np.ndarray,
    raw_scores: list[float],
    fps: float,
    title: str,
) -> None:
    frame_count = len(frames_rgb)
    raw = np.asarray(raw_scores, dtype=np.float64)
    forward_diff = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        forward_diff[:-1] = np.diff(raw)

    diff_max = max(0.05, float(np.max(np.abs(forward_diff))) if frame_count else 0.05)
    width, height = 1680, 560
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output video writer: {path}")

    for idx, frame_rgb in enumerate(frames_rgb):
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        draw_text(canvas, title[:120], (24, 36), scale=0.72, thickness=2)

        draw_text(canvas, "Original video", (24, 78), scale=0.62, thickness=2)
        resized = resize_keep_aspect_rgb(frame_rgb, 500, 390)
        x = 24 + (500 - resized.shape[1]) // 2
        y = 96 + (390 - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.rectangle(canvas, (24, 96), (524, 486), (210, 210, 210), 1)
        draw_text(canvas, f"frame {idx}/{frame_count - 1}", (24, 526), scale=0.55)

        draw_curve_panel(
            canvas,
            raw,
            idx,
            (620, 112, 1068, 456),
            "ROBOMETER raw score P[t]",
            f"current={raw[idx]:.4f}",
            0.0,
            1.0,
            (190, 45, 45),
        )
        diff_current = forward_diff[idx] if idx < len(forward_diff) else 0.0
        draw_curve_panel(
            canvas,
            forward_diff,
            idx,
            (1198, 112, 1646, 456),
            "Forward diff P[t+1] - P[t]",
            f"current={diff_current:.4f}",
            -diff_max,
            diff_max,
            (35, 120, 190),
        )
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def score_video(job: VideoJob, args: argparse.Namespace, server_url: str) -> None:
    paths = output_paths(job)
    job.output_dir.mkdir(parents=True, exist_ok=True)

    frames_rgb, fps = read_video_rgb(job.input_path)
    total_frames = int(frames_rgb.shape[0])
    request_frames_all = maybe_resize_frames(frames_rgb, args.request_max_edge)
    log(f"Video {job.split}/{job.input_path.name}: frames={total_frames}, fps={fps:.4g}")

    if not args.force:
        completed = load_completed_scores(paths["json"], total_frames)
        complete_artifacts_exist = (
            paths["csv"].exists()
            and paths["csv"].stat().st_size > 0
            and paths["mp4"].exists()
            and paths["mp4"].stat().st_size > 0
        )
        if completed is not None and complete_artifacts_exist:
            log(f"[skip] complete outputs exist for {job.input_path.name}")
            return
        if completed is not None:
            log(f"[render] complete scores exist; regenerating missing artifacts for {job.input_path.name}")
            metadata = make_metadata(
                job,
                args.task,
                fps,
                total_frames,
                args.max_frames,
                args.request_max_edge,
                server_url,
                completed,
            )
            save_json(paths["json"], metadata)
            save_scores_csv(paths["csv"], completed)
            save_score_video(paths["mp4"], frames_rgb, completed, fps, f"{job.split}: {job.input_path.stem}")
            if paths["partial"].exists():
                paths["partial"].unlink()
            log(f"[ok] wrote {paths['mp4']}")
            return

    raw_scores = [] if args.force else load_partial_scores(paths["partial"])
    if len(raw_scores) > total_frames:
        raw_scores = raw_scores[:total_frames]
    if raw_scores:
        log(f"[resume] {job.input_path.name}: starting from frame {len(raw_scores)}")

    for step in range(len(raw_scores), total_frames):
        prefix = prefix_for_step(request_frames_all, step, args.max_frames)
        score = post_progress(server_url, prefix, args.task, args.timeout, args.retries)
        raw_scores.append(float(score))
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == total_frames:
            save_json(
                paths["partial"],
                make_metadata(
                    job,
                    args.task,
                    fps,
                    total_frames,
                    args.max_frames,
                    args.request_max_edge,
                    server_url,
                    raw_scores,
                ),
            )
            log(f"  {job.input_path.name}: scored {step + 1}/{total_frames}")
        if args.sleep_between_requests > 0:
            time.sleep(args.sleep_between_requests)

    metadata = make_metadata(
        job,
        args.task,
        fps,
        total_frames,
        args.max_frames,
        args.request_max_edge,
        server_url,
        raw_scores,
    )
    save_json(paths["json"], metadata)
    save_scores_csv(paths["csv"], raw_scores)
    save_score_video(paths["mp4"], frames_rgb, raw_scores, fps, f"{job.split}: {job.input_path.stem}")
    if paths["partial"].exists():
        paths["partial"].unlink()
    log(f"[ok] wrote {paths['mp4']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--server-url", default="")
    parser.add_argument("--info-file", type=Path, default=None)
    parser.add_argument("--info-file-wait-sec", type=float, default=1800.0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--request-max-edge", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--only-split", choices=["all", "success", "unsuccess"], default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames <= 0:
        raise RuntimeError("--max-frames must be positive")
    if args.checkpoint_every <= 0:
        raise RuntimeError("--checkpoint-every must be positive")

    server_url = resolve_server_url(args)
    log(f"server_url={server_url}")
    log(f"root={args.root}")
    log(f"task={args.task}")
    log(f"max_frames={args.max_frames}")
    log(f"request_max_edge={args.request_max_edge}")
    health_check(server_url)

    jobs = build_jobs(args.root, args.only_split)
    if args.start:
        jobs = jobs[args.start :]
    if args.limit:
        jobs = jobs[: args.limit]
    log(f"jobs={len(jobs)}")
    for idx, job in enumerate(jobs, start=1):
        log(f"===== [{idx}/{len(jobs)}] {job.split}/{job.input_path.name} =====")
        score_video(job, args, server_url)
    log("all done")


if __name__ == "__main__":
    main()
