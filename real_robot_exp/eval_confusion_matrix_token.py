import h5py
import torch
# from clip_utils import normalize_embeddings
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
import io
from PIL import Image
# from eval_utils import padding_video

matplotlib.use('Agg')

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


def shorten_name(name, separator=" ", max_length=5):
    parts = name.split(separator)
    return separator.join([part[:max_length] for part in parts])

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

def plot_matrix_as_image(matrix, names, set, text, prob = False):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.1, len(matrix)))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    shortened_text = [shorten_name(name, max_length = 4) for name in text]
    shortened_names = [shorten_name(name, separator = "-", max_length = 4) for name in names]

    # Label each row and column with the given names
    ax.set_xticklabels(shortened_text, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(shortened_names, fontsize=10)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=12)
# keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    if prob:
        wandb.log({f"confusion_matrix_prob/{set}_prob_confusion_matrix": wandb.Image(fig)})
    else:
        wandb.log({f"confusion_matrix/{set}_confusion_matrix": wandb.Image(fig)})
    # plt.savefig(f"confusion_matrix_{set}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory




def plot_confusion_matrix_token(h5_file, set, self_attention_model, args, 
                        text_position_embedding = None,
                        video_position_embedding = None,
                        text_learner_parameters = None,
                        video_learner_parameters = None):
    device = next(self_attention_model.parameters()).device

    keys = list(h5_file.keys())
    if set == "train":
        eval_envs = keys[:int(len(keys)*0.5)]
    elif set == "eval":
        eval_envs = keys[int(len(keys)*0.5):]
    else:
        eval_envs = keys

    text_embeddings = []
    text_list = []
    text_mask_list = []
    for key in eval_envs:
        
        text_embedding = np.asarray(h5_file[key]["lang_embedding_individual"])
        text_embedding = torch.tensor(text_embedding).to(device).float()
        bs, seq_len, _ = text_embedding.size()
        if args.normalize_embedding:
            text_embedding = text_embedding.squeeze(0)
            text_embedding = normalize_embeddings(text_embedding)
            text_embedding = text_embedding.unsqueeze(0)
        if text_position_embedding is not None:
            text_position_embedding_add = text_position_embedding[:, :seq_len, :].to(device)
            text_embedding = text_embedding + text_position_embedding_add
        if text_learner_parameters is not None:
            text_embedding = text_learner_parameters + text_embedding.squeeze(0)
            text_embedding = text_embedding.unsqueeze(0)

        mask = torch.ones(1, seq_len + args.max_length).to(device)
        mask[:, :seq_len] = 0

        text_mask_list.append(mask)
        text_embeddings.append(text_embedding)
        text_list.append(key)
    
    predicted_progress_row = []
    if args.two_step_training:
        pred_two_step_prob_list = []
    for i in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        video_embedding = np.asarray(h5_file[env]["1"])
        video_embedding = torch.tensor(video_embedding).to(device).float()

        if args.normalize_embedding:
            traj_data = normalize_embeddings(video_embedding)
        else:
            traj_data = video_embedding

        traj_data = traj_data.unsqueeze(0)
        if args.subsample_video:
            video_embedding = padding_video(video_embedding, args.max_length)

        if video_learner_parameters is not None:
            traj_data = video_learner_parameters + video_embedding.squeeze(0)
            traj_data = video_embedding.unsqueeze(0)

        if video_position_embedding is not None:
            traj_data = traj_data + video_position_embedding




        
        text_progress_row = []
        text_prob_row = []
        for j in range(len(text_embeddings)):
            input_feature = torch.cat([text_embeddings[j], traj_data], dim=1)
            text_mask = text_mask_list[j]
            feature_len = input_feature.size(1)

            triangle_mask = torch.tril(torch.ones(feature_len, feature_len)).to(device).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
            progress_output, two_step_class = self_attention_model(input_feature, triangle_mask, text_mask)
            
            if args.catagorical_progress:
                pred_class = torch.argmax(progress_output.squeeze(0), dim = 1)
            if args.two_step_training:
                two_step_class = torch.sigmoid(two_step_class)
                two_class_label = (two_step_class >= 0.5).float().squeeze()
                text_prob_row.append(two_step_class.squeeze()[-1])
                pred_class = pred_class * two_class_label
            text_progress_row.append(pred_class[-1])
        
        text_progress_row = torch.stack(text_progress_row).detach().cpu().numpy()
        if args.two_step_training:
            pred_two_step_prob_list.append(torch.stack(text_prob_row).detach().cpu().numpy())
        predicted_progress_row.append(text_progress_row)

    predicted_progress_row = np.array(predicted_progress_row)
    if args.two_step_training:
        pred_two_step_prob_list = np.array(pred_two_step_prob_list)
        img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text_list, prob = False)
        img1 = plot_matrix_as_image(pred_two_step_prob_list, eval_envs, set, text_list, prob = True)
    else:
        img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text_list, prob = False)
    

    
