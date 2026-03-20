"""
Generate labeled H5 dataset for no-action-chunk policy training.

Reads raw data from rewind_valuemodel:
  - metaworld_generation.h5 (trajectories with state, img, action, done)
  - metaworld_embeddings_train.h5 (language embeddings)
  - reward model checkpoint (.pth)

Outputs a flat H5 with all keys needed by rewind_no-action-chunk's replay buffer:
  state, action, rewards, done, policy_lang_embedding, img_embedding, env_id

Usage (on cluster):
  cd /project2/biyik_1165/haobaizh/rewind_no-action-chunk
  python scripts/generate_labeled_dataset.py
  python scripts/generate_labeled_dataset.py --use_progress_diff --output_path datasets/metaworld_labeled_diff.h5
"""

import os
import sys
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn


# ─── ReWiNDTransformer (same architecture as trained checkpoint) ───

class ReWiNDTransformer(nn.Module):
    def __init__(self, args, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.attention_mask = nn.Transformer.generate_square_subsequent_mask(args.max_length + 1).to('cuda')

    def forward(self, video_frames, text_embed, attention_mask=None):
        batch_size = video_frames.shape[0]
        video_embed = self.video_proj(video_frames)
        text_embed = self.text_proj(text_embed).unsqueeze(1)
        video_embed[:, 0] += self.first_pos_embed
        sequence = torch.cat([text_embed, video_embed], dim=1)
        transformed = self.transformer(sequence, is_causal=True, mask=self.attention_mask)
        progress_preds = self.progress_head(transformed[:, 1:])
        return progress_preds


# ─── DINO image encoding ───

DINO_BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy-load DINO model (only when needed)
_dino_model = None

def get_dino_model():
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False).to(device)
        _dino_model.eval()
    return _dino_model


def dino_load_image(img):
    """Load and preprocess a single image for DINO."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def get_dino_embeddings(imgs_list):
    """Get DINO embeddings for a list of images (numpy arrays)."""
    dino = get_dino_model()
    episode_images_dino = [dino_load_image(img) for img in imgs_list]
    # Batch them
    batches = [
        torch.cat(episode_images_dino[i:i + DINO_BATCH_SIZE])
        for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)
    ]
    embedding_list = []
    for batch in batches:
        with torch.no_grad():
            emb = dino(batch.to(device)).squeeze().detach().cpu().numpy()
        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0)
        embedding_list.append(emb)
    return np.concatenate(embedding_list)


# ─── Utility ───

def padding_video(video_frames, max_length):
    if isinstance(video_frames, np.ndarray):
        video_frames = torch.tensor(video_frames)
    if len(video_frames) > max_length:
        index = np.linspace(0, len(video_frames) - 1, max_length).astype(int)
        video_frames = video_frames[index]
    else:
        padding_num = max_length - len(video_frames)
        last_frame = video_frames[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_num, 1)
        video_frames = torch.cat([video_frames, padding_frames], dim=0)
    return video_frames


def load_rewind_model(model_path):
    model_dict = torch.load(model_path, map_location=device, weights_only=False)
    args = model_dict["args"]
    model = ReWiNDTransformer(
        args=args, video_dim=768, text_dim=384, hidden_dim=512
    ).to(device)
    model.load_state_dict(model_dict["model_state_dict"])
    model.eval()
    return args, model


# ─── Main labeling logic ───

def label_trajectories(args, rewind_model, traj_h5, embedding_h5):
    training_keys = list(embedding_h5.keys())

    # Count total timesteps (each traj has 5 annotations)
    total_timesteps = 0
    for key in training_keys:
        for traj_id in traj_h5[key].keys():
            total_timesteps += len(traj_h5[key][traj_id]["reward"])
    total_timesteps = int(total_timesteps * 5)  # 5 annotations per trajectory

    print(f"Total timesteps to label: {total_timesteps}")

    # Create output H5 with ALL keys needed by rewind_no-action-chunk
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
        for traj_id in traj_h5[key].keys():
            traj_data = traj_h5[key][traj_id]
            num_steps = len(traj_data["done"])

            save_actions = np.array(traj_data["action"])
            save_dones = np.array(traj_data["done"])
            save_states = np.array(traj_data["state"])  # 39-dim MetaWorld state

            # Get DINO embeddings from raw images
            video_frames = np.array(traj_data["img"])
            video_frame_embeddings = get_dino_embeddings([img for img in video_frames])
            save_img_embeddings = video_frame_embeddings

            # Build progressive video slices for reward computation
            video_slices = [
                padding_video(video_frame_embeddings[0:-i], max_length=args.max_length)
                for i in range(len(video_frame_embeddings) - 1, 0, -1)
            ] + [padding_video(video_frame_embeddings[0:], max_length=args.max_length)]
            video_slices = torch.stack(video_slices).float().to(device)

            # Build mask to extract reward at the last valid frame position
            last_index_mask = torch.zeros(
                (video_slices.shape[0], args.max_length), device=device, dtype=torch.bool
            )
            for i in range(video_slices.shape[0]):
                last_frame_idx = min(i, args.max_length - 1)
                last_index_mask[i, last_frame_idx] = 1

            # Get language embeddings (5 annotations per task)
            lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])

            for i in range(len(lang_embeddings)):
                lang_emb = torch.tensor(lang_embeddings[i]).float().to(device)
                lang_batch = lang_emb.unsqueeze(0).repeat(video_slices.shape[0], 1)

                with torch.no_grad():
                    reward_outputs = rewind_model(video_slices, lang_batch).squeeze(-1)
                    reward_outputs = reward_outputs[last_index_mask]
                    progress_values = reward_outputs.cpu().numpy()

                    if args.use_progress_diff:
                        save_rewards = progress_values[1:] - progress_values[:-1]
                    else:
                        save_rewards = progress_values[1:]

                save_lang = lang_emb.repeat(num_steps, 1).cpu().numpy()

                # Write to output H5
                out["state"][current:current + num_steps] = save_states[:-1]
                out["action"][current:current + num_steps] = save_actions
                out["done"][current:current + num_steps] = save_dones
                out["rewards"][current:current + num_steps] = save_rewards
                out["policy_lang_embedding"][current:current + num_steps] = save_lang
                out["img_embedding"][current:current + num_steps] = save_img_embeddings[:-1]
                out["env_id"][current:current + num_steps] = key

                current += num_steps

    out.close()
    print(f"Done. Saved {current} timesteps to {args.output_path}")


def main():
    # Default paths assume cluster layout:
    # /project2/biyik_1165/haobaizh/
    #   rewind_valuemodel/datasets/   (source data)
    #   rewind_valuemodel/checkpoints/ (reward model)
    #   rewind_no-action-chunk/datasets/ (output)
    base = "/project2/biyik_1165/haobaizh/rewind_valuemodel"

    parser = argparse.ArgumentParser(description="Generate labeled H5 for no-action-chunk policy training.")
    parser.add_argument("--h5_video_path", default=f"{base}/datasets/metaworld_generation.h5")
    parser.add_argument("--h5_embedding_path", default=f"{base}/datasets/metaworld_embeddings_train.h5")
    parser.add_argument("--reward_model_path", default=f"{base}/checkpoints/rewind_metaworld_epoch_19.pth")
    parser.add_argument("--output_path", default="datasets/metaworld_labeled.h5")
    parser.add_argument("--use_progress_diff", action="store_true",
                        help="Use P(s')-P(s) instead of P(s) as reward.")
    args = parser.parse_args()

    config, rewind_model = load_rewind_model(args.reward_model_path)
    args.max_length = config.max_length
    print(f"Loaded ReWiND model (max_length={args.max_length})")

    traj_h5 = h5py.File(args.h5_video_path, "r")
    embedding_h5 = h5py.File(args.h5_embedding_path, "r")

    label_trajectories(args, rewind_model, traj_h5, embedding_h5)

    traj_h5.close()
    embedding_h5.close()


if __name__ == "__main__":
    main()
