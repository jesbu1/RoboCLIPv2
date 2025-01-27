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
        eval_envs = keys[:int(len(keys)*0.5)]
    elif set == "eval":
        eval_envs = keys[int(len(keys)*0.5):]
    else:
        assert False, "Invalid set"

    for key in eval_envs:
        video_group = h5_file[key]
        text_embedding = np.asarray(video_group["lang_embedding"])
        text_embedding = torch.tensor(text_embedding).to(device).float().unsqueeze(0)
        if args.normalize_embedding:
            text_embedding = normalize_embeddings(text_embedding)
        
        video_embeddings = np.asarray(video_group["1"])
        video_embeddings = torch.tensor(video_embeddings).to(device).float()
        if args.normalize_embedding:
            video_embeddings = normalize_embeddings(video_embeddings)
        if args.subsample_video:
            traj_data = sample_embedding_frames(video_embeddings, args.max_length)
        traj_data = traj_data.view(-1, 1024).unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)
        if args.cat_text_front:
            triangle_mask = torch.tril(torch.ones(traj_data.shape[1] + 1, traj_data.shape[1] + 1)).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        else:
            triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        mask = None

        pred_class, two_step_class = self_attention_model(traj_data, triangle_mask, text_embedding, mask)
        # import pdb; pdb.set_trace()
        # batch_size, seq_len, _ = traj_data.size()
        if not args.catagorical_progress:
            pred_class = pred_class.view(-1, 1)
        else:
            pred_class = torch.argmax(pred_class.squeeze(0), dim = 1).unsqueeze(1) + 1

        if args.two_step_training:
            two_step_class = two_step_class.view(-1, 2)
            two_class_label = torch.argmax(two_step_class, dim=1)
            pred_class = pred_class * two_class_label.unsqueeze(1)


        # if args.catagorical_progress:
        #     pred_class = torch.argmax(pred_class, dim = 1)
        # else:
        pred_class = pred_class.squeeze(1)

        predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())

        frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))

        figure = plt.figure()
        
        plt.plot(frame_index, predicted_classes, label="Correct Text", color="blue")
        plt.xlabel("Frame Index")
        plt.ylabel("Class")
        plt.title(f"{key}")
        if args.catagorical_progress:
            plt.ylim(-1, 6)
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


def plot_progress_token(h5_file, set, self_attention_model, args):
    device = next(self_attention_model.parameters()).device
    keys = list(h5_file.keys())
    if set == "train":
        eval_envs = keys[:int(len(keys)*0.5)]
    elif set == "eval":
        eval_envs = keys[int(len(keys)*0.5):]
    else:
        assert False, "Invalid set"

    for key in eval_envs:
        video_group = h5_file[key]
        text_embedding = np.asarray(video_group["lang_embedding_individual"])
        text_embedding = torch.tensor(text_embedding).to(device).float()
        if args.normalize_embedding:
            bs, seq_len, _ = text_embedding.size()
            text_embedding = text_embedding.view(-1, 1024)
            text_embedding = normalize_embeddings(text_embedding)
            text_embedding = text_embedding.view(bs, seq_len, 1024)

        
        video_embeddings = np.asarray(video_group["1"])
        video_embeddings = torch.tensor(video_embeddings).to(device).float()
        if args.normalize_embedding:
            video_embeddings = normalize_embeddings(video_embeddings)
        if args.subsample_video:
            traj_data = sample_embedding_frames(video_embeddings, args.max_length)
        traj_data = traj_data.view(-1, 1024).unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)
        text_len = text_embedding.shape[1]
        input_feature = torch.cat([text_embedding, traj_data], dim=1)
        triangle_mask = torch.tril(torch.ones(input_feature.shape[1], input_feature.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(input_feature.shape[0], 1, 1, 1)
        video_mask = torch.ones(1, input_feature.shape[1]).to(device)
        video_mask[:, :text_len] = 0
        progress_output, two_step_class = self_attention_model(input_feature, triangle_mask, text_len, mask = video_mask)


        # import pdb; pdb.set_trace()
        # batch_size, seq_len, _ = traj_data.size()
        pred_class = progress_output.view(-1)
        if args.two_step_training:
            two_step_class = two_step_class.view(-1, 2)
            two_class_label = torch.argmax(two_step_class, dim=1).squeeze()
            pred_class = pred_class * two_class_label
        
        batch_size, seq_len, _ = traj_data.size()
        if args.catagorical_progress:
            pred_class = torch.argmax(pred_class, dim = 1)
        else:
            pred_class = pred_class.squeeze()

        predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())

        frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))

        figure = plt.figure()
        
        plt.plot(frame_index, predicted_classes, label="Correct Text", color="blue")
        plt.xlabel("Frame Index")
        plt.ylabel("Class")
        plt.title(f"{key}")
        if args.catagorical_progress:
            plt.ylim(-1, 6)
        else:
            plt.ylim(-1, 1)
        # plt.savefig(f"progress_img/{env}.png")
        plt.legend()
        wandb.log({f"class_{set}/{key}": wandb.Image(figure)})
        plt.close()









        