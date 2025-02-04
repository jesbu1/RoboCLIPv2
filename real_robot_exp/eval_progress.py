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

def plot_progress(h5_file, set, self_attention_model, args):
    device = next(self_attention_model.parameters()).device
    keys = list(h5_file.keys())
    if set == "train":
        if args.extra_data_type == "metaworld":
            eval_envs = keys
        else:
            eval_envs = keys[:int(len(keys)*0.75)]
    elif set == "eval":
        if args.extra_data_type == "metaworld":
            eval_envs = keys
        else:
            eval_envs = keys[int(len(keys)*0.75):]
    # elif set == "test":
    #     eval_envs = keys
    else:
        assert False, "Invalid set"

    for key in tqdm(eval_envs):
        video_group = h5_file[key]
        text_embedding = np.asarray(video_group["lang_embedding"])[0].reshape(1, -1)
        text_embedding = torch.tensor(text_embedding).to(device).float().unsqueeze(0)

        if args.normalize_embedding:
            text_embedding = normalize_embeddings(text_embedding)

        video_embeddings = np.asarray(video_group["2"])
        video_embeddings = torch.tensor(video_embeddings).to(device).float()
        if args.normalize_embedding:
            video_embeddings = normalize_embeddings(video_embeddings)
        if args.subsample_video:
            traj_data = sample_embedding_frames(video_embeddings, args.max_length)
        traj_data = traj_data.view(-1, 1024).unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)
        
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1] + 1, traj_data.shape[1] + 1)).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)

        mask = None
        pred_class, two_step_class = self_attention_model(traj_data, triangle_mask, text_embedding.squeeze(1), mask)
        # batch_size, seq_len, _ = traj_data.size()
        if not args.catagorical_progress:
            pred_class = pred_class.view(-1, 1)
        else:
            if args.two_step_training:
                pred_class = torch.argmax(pred_class.squeeze(0), dim = 1).unsqueeze(1) + 1
            else:
                pred_class = torch.argmax(pred_class.squeeze(0), dim = 1).unsqueeze(1)
        pred_class = pred_class.squeeze()

        if args.two_step_training:
            two_step_class_prob = two_step_class.squeeze()
            # if two_step_class_prob > 0.5 is 1 else 0
            two_step_class_prob = two_step_class_prob > 0.5 
            two_step_class_prob = two_step_class_prob.float()

            pred_class = pred_class * two_step_class_prob


        # if args.catagorical_progress:
        #     pred_class = torch.argmax(pred_class, dim = 1)
        # else:

        predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())

        frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))

        figure = plt.figure()
        
        plt.plot(frame_index, predicted_classes, label="Correct Text", color="blue")
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

