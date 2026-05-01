#!/usr/bin/env python3
"""Label MetaWorld trajectories with TOPReward, optionally using gamma diff.

This lives in rewind_no-action-chunk so the scheduling repo can produce
TopReward datasets without editing the TopReward project scripts.
"""

import argparse
import base64
import fcntl
import glob
import io
import os
import sys
from contextlib import contextmanager

import h5py
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm


DINO_BATCH_SIZE = 64
TOPREWARD_REQUEST_FORMATS = ("chat", "raw")

ENVIRONMENT_TO_INSTRUCTION = {
    "assembly-v2": "assembly",
    "basketball-v2": "play basketball",
    "bin-picking-v2": "pick bin",
    "box-close-v2": "closing box",
    "button-press-topdown-v2": "Press the button from top",
    "button-press-topdown-wall-v2": "Press the button from top",
    "button-press-v2": "Press the button from side",
    "button-press-wall-v2": "Press the button from side",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "coffee-push-v2": "Push the coffee cup",
    "dial-turn-v2": "Turn the dial",
    "disassemble-v2": "disassemble",
    "door-close-v2": "Close the door",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "door-open-v2": "Open the door",
    "door-unlock-v2": "Turn door lock clockwise",
    "hand-insert-v2": "Pick up the block and insert it into the hole",
    "drawer-close-v2": "Close the drawer",
    "drawer-open-v2": "open drawer",
    "faucet-open-v2": "Open the faucet",
    "faucet-close-v2": "Close the faucet",
    "hammer-v2": "hammer nail",
    "handle-press-side-v2": "Press the handle from side",
    "handle-press-v2": "Press the handle",
    "handle-pull-side-v2": "Pull the handle up from the side",
    "handle-pull-v2": "Pull the handle",
    "lever-pull-v2": "pull lever",
    "peg-insert-side-v2": "Insert the peg",
    "pick-place-wall-v2": "Pick up the block and placing it to the goal position",
    "pick-out-of-hole-v2": "pick bin",
    "reach-v2": "Reach the goal",
    "push-back-v2": "Push the block back to the goal",
    "push-v2": "Push the block to the goal",
    "pick-place-v2": "Pick up the block and placing it to the goal position",
    "plate-slide-v2": "Slide the plate into the gate",
    "plate-slide-side-v2": "Slide the plate into the gate from the side",
    "plate-slide-back-v2": "Slide the plate out of the gate",
    "plate-slide-back-side-v2": "Slide the plate out of the gate from the side",
    "peg-unplug-side-v2": "unplug peg",
    "soccer-v2": "Slide the ball into the gate",
    "stick-push-v2": "Push the stick",
    "stick-pull-v2": "Pull the stick",
    "push-wall-v2": "push bin",
    "reach-wall-v2": "Reach the goal",
    "shelf-place-v2": "place bin to shelf",
    "sweep-into-v2": "Sweep the block into the hole",
    "sweep-v2": "sweep block",
    "window-open-v2": "Open the window",
    "window-close-v2": "Close the window",
}


@contextmanager
def optional_file_lock(lock_path):
    if not lock_path:
        yield
        return

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def center_crop_frame(frame, crop_size=224):
    """Match the Robometer 224x224 center-crop input budget."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3:
        return frame

    frame = frame[..., :3]
    height, width = frame.shape[:2]
    crop_height = min(crop_size, height)
    crop_width = min(crop_size, width)
    top = max((height - crop_height) // 2, 0)
    left = max((width - crop_width) // 2, 0)
    return np.ascontiguousarray(
        frame[top : top + crop_height, left : left + crop_width]
    )


def frames_to_base64(frames):
    b64_list = []
    for frame in frames:
        frame = center_crop_frame(frame)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_list


def build_topreward_prompt(instruction):
    prompt_text = (
        "The above video shows a robot manipulation trajectory "
        "that completes the following task: "
    )
    instruction_suffix = (
        f"{instruction} Decide whether the above statement is True or not. "
        "The answer is:"
    )
    return f"{prompt_text}{instruction_suffix}"


def query_raw_score_reward(
    api_url,
    model_name,
    frames_b64,
    instruction,
    lock_path,
    request_timeout,
    request_retries,
):
    payload = {
        "model": model_name,
        "frames_b64": frames_b64,
        "instruction": instruction,
    }

    last_error = None
    for attempt in range(1, request_retries + 1):
        try:
            with optional_file_lock(lock_path):
                resp = requests.post(
                    f"{api_url.rstrip('/')}/score",
                    json=payload,
                    timeout=request_timeout,
                )
                resp.raise_for_status()
                result = resp.json()
            return float(result["logprob"])
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(
                f"[TOPReward label raw] request failed on attempt "
                f"{attempt}/{request_retries}: {exc}",
                flush=True,
            )
            if attempt == request_retries:
                raise

    raise RuntimeError(f"TOPReward raw-score request failed: {last_error}")


def query_raw_score_rewards_batch(
    api_url,
    model_name,
    batch_frames_b64,
    instruction,
    lock_path,
    request_timeout,
    request_retries,
):
    payload = {
        "requests": [
            {
                "model": model_name,
                "frames_b64": frames_b64,
                "instruction": instruction,
            }
            for frames_b64 in batch_frames_b64
        ]
    }

    last_error = None
    for attempt in range(1, request_retries + 1):
        try:
            use_single_fallback = False
            with optional_file_lock(lock_path):
                resp = requests.post(
                    f"{api_url.rstrip('/')}/score_batch",
                    json=payload,
                    timeout=request_timeout,
                )
                if resp.status_code == 404:
                    use_single_fallback = True
                else:
                    resp.raise_for_status()
                    result = resp.json()
            if use_single_fallback:
                # Backward-compatible fallback for older raw servers.
                return [
                    query_raw_score_reward(
                        api_url,
                        model_name,
                        frames_b64,
                        instruction,
                        lock_path,
                        request_timeout,
                        request_retries,
                    )
                    for frames_b64 in batch_frames_b64
                ]
            return [float(item["logprob"]) for item in result["scores"]]
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(
                f"[TOPReward label raw batch] request failed on attempt "
                f"{attempt}/{request_retries}: {exc}",
                flush=True,
            )
            if attempt == request_retries:
                raise

    raise RuntimeError(f"TOPReward raw batch request failed: {last_error}")


def query_vlm_reward(
    api_url,
    model_name,
    frames_b64,
    instruction,
    lock_path,
    request_timeout,
    request_retries,
    request_format,
):
    request_format = request_format.lower()
    if request_format == "raw":
        return query_raw_score_reward(
            api_url,
            model_name,
            frames_b64,
            instruction,
            lock_path,
            request_timeout,
            request_retries,
        )
    if request_format != "chat":
        raise ValueError(
            f"Unsupported TOPReward request_format={request_format!r}; "
            f"expected one of {TOPREWARD_REQUEST_FORMATS}"
        )

    prompt = build_topreward_prompt(instruction)

    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        for b64 in frames_b64
    ]
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 5,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }

    last_error = None
    for attempt in range(1, request_retries + 1):
        try:
            with optional_file_lock(lock_path):
                resp = requests.post(
                    f"{api_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    timeout=request_timeout,
                )
                resp.raise_for_status()
                result = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(
                f"[TOPReward label chat] request failed on attempt "
                f"{attempt}/{request_retries}: {exc}",
                flush=True,
            )
            if attempt == request_retries:
                raise
    else:
        raise RuntimeError(f"TOPReward chat request failed: {last_error}")

    logprobs_content = result["choices"][0]["logprobs"]["content"]
    if logprobs_content:
        first_token = logprobs_content[0]
        for entry in first_token.get("top_logprobs", []):
            if entry["token"].strip().lower() == "true":
                return entry["logprob"]
        if first_token["token"].strip().lower() == "true":
            return first_token["logprob"]

    return -10.0


def query_vlm_reward_legacy(
    api_url,
    model_name,
    frames_b64,
    instruction,
    lock_path,
    request_timeout,
    request_retries,
):
    return query_vlm_reward(
        api_url,
        model_name,
        frames_b64,
        instruction,
        lock_path,
        request_timeout,
        request_retries,
        "chat",
    )


def compute_prefix_rewards(
    api_url,
    model_name,
    video_frames,
    instruction,
    max_frames_per_query,
    lock_path,
    request_timeout,
    request_retries,
    request_format,
    request_batch_size,
):
    num_frames = len(video_frames)
    max_frames_per_query = min(max_frames_per_query, num_frames)

    prefix_batches = []
    for length in range(1, num_frames + 1):
        prefix = video_frames[:length]
        if len(prefix) > max_frames_per_query:
            indices = np.linspace(
                0,
                len(prefix) - 1,
                max_frames_per_query,
                dtype=int,
            )
            prefix = [prefix[i] for i in indices]
        prefix_batches.append(frames_to_base64(prefix))

    if request_format.lower() == "raw":
        prefix_rewards = []
        request_batch_size = max(int(request_batch_size), 1)
        for start in range(0, len(prefix_batches), request_batch_size):
            chunk = prefix_batches[start : start + request_batch_size]
            prefix_rewards.extend(
                query_raw_score_rewards_batch(
                    api_url,
                    model_name,
                    chunk,
                    instruction,
                    lock_path,
                    request_timeout,
                    request_retries,
                )
            )
        return np.asarray(prefix_rewards, dtype=np.float32)

    prefix_rewards = []
    for b64 in prefix_batches:
        prefix_rewards.append(
            query_vlm_reward(
                api_url,
                model_name,
                b64,
                instruction,
                lock_path,
                request_timeout,
                request_retries,
                request_format,
            )
        )

    return np.asarray(prefix_rewards, dtype=np.float32)


def load_topreward_helpers(topreward_dir):
    sys.path.insert(0, topreward_dir)

    from utils.processing_utils import dino_load_image

    return ENVIRONMENT_TO_INSTRUCTION, dino_load_image


def _unique_existing_order(paths):
    seen = set()
    unique = []
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.path.expanduser(path))
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def load_dinov2_vitb14_offline(device):
    """Load DINOv2 from an existing torch hub cache without touching the network."""

    user = os.environ.get("USER", "haobaizh")
    torch_homes = _unique_existing_order(
        [
            os.environ.get("TORCH_HOME"),
            f"/home1/{user}/.cache/torch",
            "/home1/haobaizh/.cache/torch",
            f"/scratch1/{user}/.cache/torch",
            "/scratch1/haobaizh/.cache/torch",
        ]
    )

    candidates = []
    explicit_repo = os.environ.get("DINOV2_REPO_DIR")
    if explicit_repo:
        candidates.append((os.path.abspath(os.path.expanduser(explicit_repo)), os.environ.get("TORCH_HOME")))
    for torch_home in torch_homes:
        candidates.append(
            (
                os.path.join(torch_home, "hub", "facebookresearch_dinov2_main"),
                torch_home,
            )
        )

    checked = []
    for repo_dir, torch_home in candidates:
        if not torch_home:
            torch_home = os.path.dirname(os.path.dirname(repo_dir))
        torch_home = os.path.abspath(os.path.expanduser(torch_home))
        hubconf = os.path.join(repo_dir, "hubconf.py")
        ckpts = sorted(
            glob.glob(os.path.join(torch_home, "hub", "checkpoints", "dinov2_vitb14*.pth"))
        )
        checked.append(f"repo={repo_dir} hubconf={os.path.isfile(hubconf)} ckpts={len(ckpts)}")
        if os.path.isfile(hubconf) and ckpts:
            os.environ["TORCH_HOME"] = torch_home
            print(f"Loading DINOv2 from local repo: {repo_dir}", flush=True)
            print(f"Using TORCH_HOME={torch_home}", flush=True)
            print(f"Using checkpoint={ckpts[0]}", flush=True)
            return torch.hub.load(
                repo_dir,
                "dinov2_vitb14",
                source="local",
                force_reload=False,
            ).to(device)

    raise RuntimeError(
        "DINOv2 local cache not found, and label jobs are not allowed to fetch it "
        "from GitHub at runtime. Checked:\n  " + "\n  ".join(checked)
    )


def get_dino_embeddings(dinov2_vits14, dino_load_image, imgs_list, device):
    episode_images_dino = [dino_load_image(img) for img in imgs_list]
    episode_images_dino = [
        torch.concatenate(episode_images_dino[i : i + DINO_BATCH_SIZE])
        for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)
    ]

    embedding_list = []
    with torch.inference_mode():
        for batch in episode_images_dino:
            embeddings = dinov2_vits14(batch.to(device)).squeeze().detach().cpu().numpy()
            if len(embeddings.shape) == 1:
                embeddings = np.expand_dims(embeddings, 0)
            embedding_list.append(embeddings)
    return np.concatenate(embedding_list)


def get_output_specs(args):
    if args.baseline_output_path or args.diff_output_path:
        specs = []
        if args.baseline_output_path:
            specs.append(("baseline", args.baseline_output_path, args.baseline_reward_scale))
        if args.diff_output_path:
            specs.append(("diff", args.diff_output_path, args.diff_reward_scale))
        return specs

    if not args.output_path:
        raise ValueError(
            "Either --output_path or --baseline_output_path/--diff_output_path must be set."
        )
    if not args.mode:
        raise ValueError("--mode is required when writing a single --output_path.")
    return [(args.mode, args.output_path, args.reward_scale)]


def create_labeled_dataset(handle, total_timesteps):
    handle.create_dataset("state", (total_timesteps, 39), dtype="float32")
    handle.create_dataset("action", (total_timesteps, 4), dtype="float32")
    handle.create_dataset("rewards", (total_timesteps,), dtype="float32")
    handle.create_dataset("done", (total_timesteps,), dtype="float32")
    handle.create_dataset(
        "policy_lang_embedding", (total_timesteps, 384), dtype="float32"
    )
    handle.create_dataset("img_embedding", (total_timesteps, 768), dtype="float32")
    handle.create_dataset("env_id", (total_timesteps,), dtype="S20")


def write_labeled_slice(
    dataset,
    sl,
    save_states,
    save_actions,
    save_dones,
    save_video_slices,
    save_reward_outputs,
    lang_embedding,
    key,
):
    num_steps = len(save_actions)
    dataset["state"][sl] = save_states
    dataset["action"][sl] = save_actions
    dataset["done"][sl] = save_dones
    dataset["rewards"][sl] = save_reward_outputs
    dataset["policy_lang_embedding"][sl] = np.tile(lang_embedding, (num_steps, 1))
    dataset["img_embedding"][sl] = save_video_slices[:-1]
    dataset["env_id"][sl] = key


def label_trajectories(args):
    environment_to_instruction, dino_load_image = load_topreward_helpers(args.topreward_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vits14 = load_dinov2_vitb14_offline(device)
    output_specs = get_output_specs(args)

    with h5py.File(args.h5_video_path, "r") as traj_h5, h5py.File(
        args.h5_embedding_path, "r"
    ) as embedding_h5:
        training_keys = list(embedding_h5.keys())
        if args.num_task_shards < 1:
            raise ValueError("--num_task_shards must be >= 1")
        if not 0 <= args.task_shard_index < args.num_task_shards:
            raise ValueError("--task_shard_index must satisfy 0 <= index < num_task_shards")
        selected_keys = training_keys[args.task_shard_index :: args.num_task_shards]
        if not selected_keys:
            raise ValueError(
                f"No tasks selected for shard {args.task_shard_index}/{args.num_task_shards}"
            )

        total_timesteps = 0
        for key in selected_keys:
            task_num_annotations = len(
                np.array(embedding_h5[key]["minilm_lang_embedding"])
            )
            for traj_id in traj_h5[key].keys():
                total_timesteps += (
                    len(traj_h5[key][traj_id]["reward"]) * task_num_annotations
                )

        handles = {}
        try:
            for mode, path, _scale in output_specs:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                handle = h5py.File(path, "w")
                create_labeled_dataset(handle, total_timesteps)
                handles[mode] = handle

            print(
                f"Selected {len(selected_keys)}/{len(training_keys)} tasks "
                f"for shard {args.task_shard_index}/{args.num_task_shards}: "
                f"{', '.join(selected_keys)}",
                flush=True,
            )
            current_timestep = 0
            for key in tqdm(selected_keys):
                instruction = environment_to_instruction.get(key, key)

                for traj_id in traj_h5[key].keys():
                    traj_data = traj_h5[key][traj_id]
                    num_steps = len(traj_data["done"])
                    video_frames = np.array(traj_data["img"])
                    video_frames_list = [img for img in video_frames]

                    save_actions = np.array(traj_data["action"])
                    save_dones = np.array(traj_data["done"])
                    save_states = np.array(traj_data["state"])
                    save_video_slices = get_dino_embeddings(
                        dinov2_vits14, dino_load_image, video_frames_list, device
                    )

                    # Match Robometer labeling: query every prefix step, but
                    # cap each request to a small uniform sample of frames.
                    per_step_rewards = compute_prefix_rewards(
                        args.api_url,
                        args.model_name,
                        video_frames_list,
                        instruction,
                        args.num_prefix_samples,
                        args.lock_path,
                        args.request_timeout,
                        args.request_retries,
                        args.request_format,
                        args.request_batch_size,
                    )

                    reward_outputs_by_mode = {}
                    for mode, _path, reward_scale in output_specs:
                        if mode == "diff":
                            if args.use_reverse_progress_diff:
                                raw_outputs = (
                                    per_step_rewards[1:]
                                    - args.diff_gamma * per_step_rewards[:-1]
                                )
                            else:
                                raw_outputs = (
                                    args.diff_gamma * per_step_rewards[1:]
                                    - per_step_rewards[:-1]
                                )
                        else:
                            raw_outputs = per_step_rewards[1:]
                        save_reward_outputs = reward_scale * raw_outputs
                        if len(save_reward_outputs) != num_steps:
                            raise RuntimeError(
                                f"Reward length mismatch for {key}/{traj_id}/{mode}: "
                                f"{len(save_reward_outputs)} vs {num_steps}"
                            )
                        reward_outputs_by_mode[mode] = save_reward_outputs

                    lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])
                    for lang_embedding in lang_embeddings:
                        sl = slice(current_timestep, current_timestep + num_steps)
                        for mode, dataset in handles.items():
                            write_labeled_slice(
                                dataset,
                                sl,
                                save_states,
                                save_actions,
                                save_dones,
                                save_video_slices,
                                reward_outputs_by_mode[mode],
                                lang_embedding,
                                key,
                            )
                        current_timestep += num_steps

            print(f"Successfully processed and saved {current_timestep} timesteps.")
        finally:
            for handle in handles.values():
                handle.close()


def main():
    parser = argparse.ArgumentParser(description="Label rewards using TOPReward.")
    parser.add_argument("--topreward_dir", default=os.environ.get("TOPREWARD_DIR", "/scratch1/haobaizh/rewind_topreward"))
    parser.add_argument("--h5_video_path", required=True)
    parser.add_argument("--h5_embedding_path", required=True)
    parser.add_argument("--output_path")
    parser.add_argument(
        "--baseline_output_path",
        help="Optional H5 path for baseline raw-score rewards. If set with "
        "--diff_output_path, both are written from one TOPReward pass.",
    )
    parser.add_argument(
        "--diff_output_path",
        help="Optional H5 path for diff rewards. If set with "
        "--baseline_output_path, both are written from one TOPReward pass.",
    )
    parser.add_argument("--api_url", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num_prefix_samples", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--request_retries", type=int, default=2)
    parser.add_argument("--request_batch_size", type=int, default=8)
    parser.add_argument(
        "--request_format",
        choices=TOPREWARD_REQUEST_FORMATS,
        default="chat",
        help="chat uses vLLM /v1/chat/completions; raw uses the custom /score endpoint without a chat template.",
    )
    parser.add_argument("--mode", choices=["baseline", "diff"])
    parser.add_argument("--use_reverse_progress_diff", action="store_true")
    parser.add_argument("--diff_gamma", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--baseline_reward_scale", type=float, default=1.0)
    parser.add_argument("--diff_reward_scale", type=float, default=1.0)
    parser.add_argument("--lock_path", default="")
    parser.add_argument("--task_shard_index", type=int, default=0)
    parser.add_argument("--num_task_shards", type=int, default=1)
    args = parser.parse_args()

    label_trajectories(args)


if __name__ == "__main__":
    main()
