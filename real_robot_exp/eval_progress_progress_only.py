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
from scipy.stats import spearmanr, pearsonr


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

def plot_progress(h5_file, set, self_attention_model, args, epoch = 0):
    device = next(self_attention_model.parameters()).device
    keys = list(h5_file.keys())
    eval_envs = keys
    
    # 存储所有环境的相关系数
    all_env_spearman = []
    all_env_pearson = []

    for key in tqdm(eval_envs):
        video_group = h5_file[key]

        if args.text_embedding_model == "minilm":
            text_embedding = np.asarray(video_group["minilm_lang_embedding"])[0].reshape(1, -1)
        else:
            text_embedding = np.asarray(video_group["liv_lang_embedding"])[0].reshape(1, -1)
        text_embedding = torch.from_numpy(text_embedding).to(device).float()

        if args.normalize_embedding:
            text_embedding = normalize_embeddings(text_embedding)
        
        # Get all trajectory keys (exclude language embeddings)
        traj_keys = [k for k in video_group.keys() if "lang" not in k]
        random.shuffle(traj_keys)
        traj_keys = traj_keys[:5]
        figure = plt.figure(figsize=(10, 6))
        
        # Plot each trajectory with a different color
        colors = plt.cm.rainbow(np.linspace(0, 1, len(traj_keys)))
        
        # 存储当前环境的相关系数
        env_spearmans = []
        env_pearsons = []
        
        for traj_idx, traj_key in enumerate(traj_keys):
            video_embeddings = np.asarray(video_group[traj_key])
            video_embeddings = torch.from_numpy(video_embeddings).to(device).float()

            if args.normalize_embedding:
                video_embeddings = normalize_embeddings(video_embeddings)
            if args.subsample_video:
                traj_data = sample_embedding_frames(video_embeddings, args.max_length)
            else:
                traj_data = video_embeddings
            
            traj_data = traj_data.view(-1, 768).unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)

            pred_class, two_step_class = self_attention_model(traj_data, text_embedding)

            pred_class = pred_class 

            predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
            predicted_classes = predicted_classes[1:]  # Remove first frame prediction

            frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))
            
            # 创建0到1的直线作为GT
            gt_array = np.linspace(0, 1, len(predicted_classes))
            
            # 计算Spearman和Pearson相关系数
            spearman_coef, _ = spearmanr(predicted_classes, gt_array)
            pearson_coef, _ = pearsonr(predicted_classes, gt_array)
            
            env_spearmans.append(spearman_coef)
            env_pearsons.append(pearson_coef)
            
            plt.plot(frame_index, predicted_classes, label=f"{traj_idx+1}", color=colors[traj_idx], alpha=0.7)
        
        # 计算当前环境的平均相关系数
        if env_spearmans:  # 确保不是空列表
            avg_env_spearman = np.mean(env_spearmans)
            avg_env_pearson = np.mean(env_pearsons)
            all_env_spearman.append(avg_env_spearman)
            all_env_pearson.append(avg_env_pearson)
    
    # 计算所有环境的平均相关系数
    if all_env_spearman:  # 确保不是空列表
        overall_avg_spearman = np.mean(all_env_spearman)
        overall_avg_pearson = np.mean(all_env_pearson)
        
        # 上传到wandb
        wandb.log({
            f"{set}_avg_spearman_correlation": overall_avg_spearman,
            f"{set}_avg_pearson_correlation": overall_avg_pearson,
            "epoch": epoch
        })

    plt.xlabel("Frame Index")
    plt.ylabel("Progress")
    plt.title(f"{key}")
    if args.catagorical_progress:
        plt.ylim(-1, 12)
    else:
        plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    wandb.log({f"class_{set}/{key}": wandb.Image(figure, caption=f"Epoch {epoch}")})
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

