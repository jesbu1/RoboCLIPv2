import h5py
import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb
import matplotlib
from tqdm import tqdm
import imageio
# from animation_utils import animate_video_with_rewards, log_gif_to_wandb, compute_mmrv, animate_video_with_rewards_class
import torch.nn.functional as F
import copy


matplotlib.use('Agg')


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

def padding_video(video_frames, max_length):
    video_length = len(video_frames)
    if type(video_frames) == np.ndarray:
        video_frames = torch.tensor(video_frames)
    if video_length < max_length:
        # padding first frame
        padding_length = max_length - video_length
        first_frame = video_frames[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_length, 1)
        video_frames = torch.cat([padding_frames, video_frames], dim=0)
    
    elif video_length > max_length:
        frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
        video_frames = video_frames[frame_idx]

    return video_frames

def sample_video_frames(frames, num_frames = 32):
    total_frames = len(frames)
    if total_frames > num_frames:
        # sample num_frames frames
        index = np.linspace(0, total_frames-1, num_frames).astype(int)
        frames = frames[index]

    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        frames = [frames[0]] * padding_num + frames

    return frames

def plot_progress(h5_file, set, video_encoder, text_encoder, args, pca_text_model = None, pca_video_model = None):
    device = next(video_encoder.parameters()).device
    keys = list(h5_file.keys())
    eval_envs = keys


    for key in tqdm(eval_envs):
        video_group = h5_file[key]
        select_key = list(video_group.keys())
        select_key = [k for k in select_key if "liv_lang_embedding_individual" in k]
        select_key = random.choice(select_key)

        text_embedding = np.asarray(video_group[select_key])[0]
        text_embedding = torch.from_numpy(text_embedding).to(device).float().unsqueeze(0)

        if args.pca:
            text_embedding = pca_text_model(text_embedding)
        if args.normalize_embedding:
            text_embedding = normalize_embeddings(text_embedding)

        choose_key = list(video_group.keys())
        choose_key = [k for k in choose_key if "lang" not in k]
        choose_key = random.choice(choose_key)
        video_embeddings = np.asarray(video_group[choose_key])
        video_embeddings = torch.from_numpy(video_embeddings).to(device).float()
        if args.pca:
            video_embeddings = pca_video_model(video_embeddings)
        if args.normalize_embedding:
            video_embeddings = normalize_embeddings(video_embeddings)
        if args.subsample_video:
            traj_data = sample_embedding_frames(video_embeddings, args.max_length)
        
        if args.pca:
            traj_data = traj_data.view(-1, 512).unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)
        else:
            traj_data = traj_data.view(-1, 768).unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)

        
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        video_embeddings = video_encoder(traj_data, triangle_mask).squeeze(0)
        text_mask = torch.zeros(text_embedding.shape[0], text_embedding.shape[1]).to(device).bool()
        
        text_embedding = text_encoder(text_embedding, text_mask)
        
        if args.norm_length:
            video_embeddings = F.normalize(video_embeddings, p=2, dim=1)
            text_embedding = F.normalize(text_embedding, p=2, dim=1)
        text_embedding = text_embedding.unsqueeze(0)
        text_embedding = text_embedding.repeat(1, video_embeddings.shape[0], 1).squeeze(0)

        progress = torch.sum(video_embeddings * text_embedding, dim=-1).detach().cpu().numpy()

        frame_index = np.linspace(1, len(progress), len(progress))

        figure = plt.figure()
        
        plt.plot(frame_index, progress, label="Correct Text", color="blue")
        plt.xlabel("Frame Index")
        plt.ylabel("Class")
        plt.title(f"{key}")
        if args.catagorical_progress:
            plt.ylim(-1, 12)
        else:
            plt.ylim(-1, 1)
        # plt.savefig(f"progress_img/{env}.png")
        plt.legend()
        wandb.log({f"class_{set}/{key}": wandb.Image(figure)})
        plt.close()

def sample_video_frames(frames, num_frames = 32):
    total_frames = len(frames)
    if total_frames > num_frames:
        frames = frames[::total_frames//num_frames]
    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        frames = [frames[0]] * padding_num + frames

    return frames

def sample_embedding_frames(embeddings, num_frames = 32):
    total_frames = embeddings.shape[0]
    if total_frames > num_frames:
        index = np.linspace(0, total_frames-1, num_frames).astype(int)
        embeddings = embeddings[index]

    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        first_frame = embeddings[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_num, 1)
        embeddings = torch.cat([padding_frames, embeddings], dim=0)
    return embeddings

