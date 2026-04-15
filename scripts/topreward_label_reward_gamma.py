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


def query_vlm_reward(
    api_url,
    model_name,
    frames_b64,
    instruction,
    lock_path,
    request_timeout,
    request_retries,
):
    prompt_text = (
        "The above video shows a robot manipulation trajectory "
        "that completes the following task: "
    )
    instruction_suffix = (
        f"{instruction} Decide whether the above statement is True or not. "
        "The answer is:"
    )

    content = []
    for b64 in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": f"{prompt_text}{instruction_suffix}"})

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
                f"[TOPReward label] request failed on attempt "
                f"{attempt}/{request_retries}: {exc}",
                flush=True,
            )
            if attempt == request_retries:
                raise
    else:
        raise RuntimeError(f"TOPReward request failed: {last_error}")

    logprobs_content = result["choices"][0]["logprobs"]["content"]
    if logprobs_content:
        first_token = logprobs_content[0]
        for entry in first_token.get("top_logprobs", []):
            if entry["token"].strip().lower() == "true":
                return entry["logprob"]
        if first_token["token"].strip().lower() == "true":
            return first_token["logprob"]

    return -10.0


def compute_prefix_rewards(
    api_url,
    model_name,
    video_frames,
    instruction,
    num_samples,
    lock_path,
    request_timeout,
    request_retries,
):
    num_frames = len(video_frames)
    num_samples = min(num_samples, num_frames)

    if num_frames > 2:
        prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
        prefix_lengths = sorted(set(int(x) for x in prefix_lengths))
    else:
        prefix_lengths = [num_frames]

    prefix_rewards = []
    for length in prefix_lengths:
        prefix = video_frames[:length]
        if len(prefix) > num_samples:
            indices = np.linspace(0, len(prefix) - 1, num_samples, dtype=int)
            prefix = [prefix[i] for i in indices]
        b64 = frames_to_base64(prefix)
        prefix_rewards.append(
            query_vlm_reward(
                api_url,
                model_name,
                b64,
                instruction,
                lock_path,
                request_timeout,
                request_retries,
            )
        )

    all_steps = np.arange(1, num_frames + 1)
    return np.interp(all_steps, prefix_lengths, prefix_rewards)


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


def label_trajectories(args):
    environment_to_instruction, dino_load_image = load_topreward_helpers(args.topreward_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vits14 = load_dinov2_vitb14_offline(device)

    with h5py.File(args.h5_video_path, "r") as traj_h5, h5py.File(
        args.h5_embedding_path, "r"
    ) as embedding_h5:
        training_keys = list(embedding_h5.keys())

        total_timesteps = 0
        for key in training_keys:
            task_num_annotations = len(
                np.array(embedding_h5[key]["minilm_lang_embedding"])
            )
            for traj_id in traj_h5[key].keys():
                total_timesteps += (
                    len(traj_h5[key][traj_id]["reward"]) * task_num_annotations
                )

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with h5py.File(args.output_path, "w") as labeled_dataset:
            labeled_dataset.create_dataset("state", (total_timesteps, 39), dtype="float32")
            labeled_dataset.create_dataset("action", (total_timesteps, 4), dtype="float32")
            labeled_dataset.create_dataset("rewards", (total_timesteps,), dtype="float32")
            labeled_dataset.create_dataset("done", (total_timesteps,), dtype="float32")
            labeled_dataset.create_dataset(
                "policy_lang_embedding", (total_timesteps, 384), dtype="float32"
            )
            labeled_dataset.create_dataset(
                "img_embedding", (total_timesteps, 768), dtype="float32"
            )
            labeled_dataset.create_dataset("env_id", (total_timesteps,), dtype="S20")

            current_timestep = 0
            for key in tqdm(training_keys):
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

                    per_step_rewards = compute_prefix_rewards(
                        args.api_url,
                        args.model_name,
                        video_frames_list,
                        instruction,
                        args.num_prefix_samples,
                        args.lock_path,
                        args.request_timeout,
                        args.request_retries,
                    )

                    if args.mode == "diff":
                        save_reward_outputs = (
                            args.reward_scale
                            * (args.diff_gamma * per_step_rewards[1:] - per_step_rewards[:-1])
                        )
                    else:
                        save_reward_outputs = args.reward_scale * per_step_rewards[1:]

                    if len(save_reward_outputs) != num_steps:
                        raise RuntimeError(
                            f"Reward length mismatch for {key}/{traj_id}: "
                            f"{len(save_reward_outputs)} vs {num_steps}"
                        )

                    lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])
                    for lang_embedding in lang_embeddings:
                        sl = slice(current_timestep, current_timestep + num_steps)
                        labeled_dataset["state"][sl] = save_states
                        labeled_dataset["action"][sl] = save_actions
                        labeled_dataset["done"][sl] = save_dones
                        labeled_dataset["rewards"][sl] = save_reward_outputs
                        labeled_dataset["policy_lang_embedding"][sl] = np.tile(
                            lang_embedding, (num_steps, 1)
                        )
                        labeled_dataset["img_embedding"][sl] = save_video_slices[:-1]
                        labeled_dataset["env_id"][sl] = key
                        current_timestep += num_steps

            print(f"Successfully processed and saved {current_timestep} timesteps.")


def main():
    parser = argparse.ArgumentParser(description="Label rewards using TOPReward.")
    parser.add_argument("--topreward_dir", default=os.environ.get("TOPREWARD_DIR", "/scratch1/haobaizh/rewind_topreward"))
    parser.add_argument("--h5_video_path", required=True)
    parser.add_argument("--h5_embedding_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--api_url", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num_prefix_samples", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--request_retries", type=int, default=2)
    parser.add_argument("--mode", choices=["baseline", "diff"], required=True)
    parser.add_argument("--diff_gamma", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--lock_path", default="")
    args = parser.parse_args()

    label_trajectories(args)


if __name__ == "__main__":
    main()
