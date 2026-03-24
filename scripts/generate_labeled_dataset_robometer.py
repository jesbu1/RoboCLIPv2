"""
Label MetaWorld trajectories with Robometer progress rewards via HTTP server.
Same H5 structure as generate_labeled_dataset.py but rewards from Robometer.
DINO embeddings still computed for policy observations.

Usage:
  python scripts/generate_labeled_dataset_robometer.py --server_url http://node:8000
  python scripts/generate_labeled_dataset_robometer.py --server_url http://node:8000 \
      --use_progress_diff --diff_gamma 0.99 --output_path datasets/metaworld_labeled_robometer_diff_gamma099.h5
"""

import os
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import requests
import io
import base64

# ─── Task text mapping ───
ENVIRONMENT_TO_INSTRUCTION = {
    "button-press-topdown-wall-v2": "Press the button from top",
    "button-press-v2": "Press the button from side",
    "button-press-wall-v2": "Press the button from side",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "coffee-push-v2": "Push the coffee cup",
    "dial-turn-v2": "Turn the dial",
    "door-close-v2": "Close the door",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "door-open-v2": "Open the door",
    "door-unlock-v2": "Turn door lock clockwise",
    "drawer-close-v2": "Close the drawer",
    "faucet-open-v2": "Open the faucet",
    "faucet-close-v2": "Close the faucet",
    "handle-press-side-v2": "Press the handle from side",
    "handle-press-v2": "Press the handle",
    "handle-pull-side-v2": "Pull the handle up from the side",
    "handle-pull-v2": "Pull the handle",
    "hand-insert-v2": "Pick up the block and insert it into the hole",
    "peg-insert-side-v2": "Insert the peg",
    "pick-place-v2": "Pick up the block and placing it to the goal position",
    "plate-slide-v2": "Slide the plate into the gate",
    "plate-slide-back-side-v2": "Slide the plate out of the gate from the side",
    "push-v2": "Push the block to the goal",
    "reach-v2": "Reach the goal",
    "reach-wall-v2": "Reach the goal",
    "stick-push-v2": "Push the stick",
    "stick-pull-v2": "Pull the stick",
    "sweep-into-v2": "Sweep the block into the hole",
    "window-open-v2": "Open the window",
    "window-close-v2": "Close the window",
}

# ─── DINO (same preprocessing as reward model training + online wrapper) ───
DINO_BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dino_model = None

dino_transform = T.Compose([
    T.ToTensor(), T.CenterCrop(224), T.Normalize([0.5], [0.5]),
])

def get_dino_model():
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False).to(device)
        _dino_model.eval()
    return _dino_model

def dino_load_image(img):
    img = Image.fromarray(img)
    return dino_transform(img)[:3].unsqueeze(0)

def get_dino_embeddings(imgs_list):
    dino = get_dino_model()
    tensors = [dino_load_image(img) for img in imgs_list]
    batches = [torch.cat(tensors[i:i+DINO_BATCH_SIZE]) for i in range(0, len(tensors), DINO_BATCH_SIZE)]
    embs = []
    for batch in batches:
        with torch.no_grad():
            e = dino(batch.to(device)).squeeze().detach().cpu().numpy()
        if e.ndim == 1:
            e = np.expand_dims(e, 0)
        embs.append(e)
    return np.concatenate(embs)

# ─── Robometer server call ───
def robometer_progress_per_step(frames, task_text, server_url, max_frames=4):
    F = frames.shape[0]
    progress = np.zeros(F, dtype=np.float32)
    fail_count = 0
    for t in range(F):
        sub = frames[:t+1]
        if sub.shape[0] > max_frames:
            idx = np.linspace(0, sub.shape[0]-1, max_frames, dtype=int)
            sub = sub[idx]
        buf = io.BytesIO()
        np.save(buf, sub)
        payload = {
            "frames_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "task": task_text,
            "sample_type": "progress",
        }
        try:
            resp = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            p = result.get("progress", result.get("reward", 0.0))
            progress[t] = float(p[-1]) if isinstance(p, list) else float(p)
        except Exception as e:
            fail_count += 1
            print(f"  [WARNING] Server failed at step {t}: {e}")
            if fail_count >= 10:
                raise RuntimeError(f"Server failed {fail_count} times")
    return progress

# ─── Main ───
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_video_path", default="/project2/biyik_1165/haobaizh/rewind_valuemodel/datasets/metaworld_generation.h5")
    parser.add_argument("--h5_embedding_path", default="/project2/biyik_1165/haobaizh/rewind_valuemodel/datasets/metaworld_embeddings_train.h5")
    parser.add_argument("--output_path", default="datasets/metaworld_labeled_robometer.h5")
    parser.add_argument("--server_url", default="http://localhost:8000")
    parser.add_argument("--max_frames", type=int, default=4)
    parser.add_argument("--use_progress_diff", action="store_true")
    parser.add_argument("--diff_gamma", type=float, default=1.0)
    args = parser.parse_args()

    traj_h5 = h5py.File(args.h5_video_path, "r")
    embedding_h5 = h5py.File(args.h5_embedding_path, "r")
    training_keys = list(embedding_h5.keys())

    total_timesteps = 0
    for key in training_keys:
        for traj_id in traj_h5[key].keys():
            total_timesteps += len(traj_h5[key][traj_id]["reward"])
    num_ann = len(np.array(embedding_h5[training_keys[0]]["minilm_lang_embedding"]))
    total_timesteps = int(total_timesteps * num_ann)

    print(f"Total timesteps: {total_timesteps}")
    print(f"Mode: {'diff gamma={}'.format(args.diff_gamma) if args.use_progress_diff else 'baseline P(s)'}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out = h5py.File(args.output_path, "w")
    out.create_dataset("state", (total_timesteps, 39), dtype="float32")
    out.create_dataset("action", (total_timesteps, 4), dtype="float32")
    out.create_dataset("rewards", (total_timesteps,), dtype="float32")
    out.create_dataset("done", (total_timesteps,), dtype="float32")
    out.create_dataset("policy_lang_embedding", (total_timesteps, 384), dtype="float32")
    out.create_dataset("img_embedding", (total_timesteps, 768), dtype="float32")
    out.create_dataset("env_id", (total_timesteps,), dtype="S20")

    current = 0
    for key in tqdm(training_keys, desc="Tasks"):
        task_text = ENVIRONMENT_TO_INSTRUCTION.get(key, key)
        lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])

        for traj_id in tqdm(list(traj_h5[key].keys()), desc=f"  {key}", leave=False):
            traj_data = traj_h5[key][traj_id]
            num_steps = len(traj_data["done"])
            save_actions = np.array(traj_data["action"])
            save_dones = np.array(traj_data["done"])
            save_states = np.array(traj_data["state"])
            video_frames = np.array(traj_data["img"])

            dino_embs = get_dino_embeddings([img for img in video_frames])
            save_img_embs = dino_embs[:-1]

            progress_values = robometer_progress_per_step(
                video_frames, task_text, args.server_url, args.max_frames
            )

            if args.use_progress_diff:
                save_rewards = args.diff_gamma * progress_values[1:] - progress_values[:-1]
            else:
                save_rewards = progress_values[1:]

            for i in range(len(lang_embeddings)):
                end = current + num_steps
                out["state"][current:end] = save_states
                out["action"][current:end] = save_actions
                out["rewards"][current:end] = save_rewards
                out["done"][current:end] = save_dones
                out["policy_lang_embedding"][current:end] = np.tile(lang_embeddings[i], (num_steps, 1))
                out["img_embedding"][current:end] = save_img_embs
                out["env_id"][current:end] = key
                current += num_steps

    out.close()
    print(f"Done. Saved {current} timesteps to {args.output_path}")

if __name__ == "__main__":
    main()
