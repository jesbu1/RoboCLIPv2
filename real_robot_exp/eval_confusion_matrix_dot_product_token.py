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






def plot_confusion_matrix(h5_file, set, video_encoder, text_encoder, args, pca_text_model = None, pca_video_model = None):
    device = next(video_encoder.parameters()).device

    keys = list(h5_file.keys())
    eval_envs = keys


    text_embeddings = []
    text_list = []
    for key in eval_envs:
        video_group = h5_file[key]
        select_key = list(video_group.keys())
        select_key = [k for k in select_key if "liv_lang_embedding_individual" in k]
        select_key = random.choice(select_key)
        embedding = np.asarray(video_group[select_key])
        embedding = torch.from_numpy(embedding).to(device).float()
        text_mask = torch.zeros(embedding.shape[0], embedding.shape[1]).to(device).bool()
        embedding = text_encoder(embedding, text_mask).detach().cpu().numpy()
        text_embeddings.append(embedding)
        text_list.append(key)

    text_embeddings = np.concatenate(text_embeddings, axis=0)
    text_embeddings = torch.from_numpy(text_embeddings).to(device).float()
    if args.pca:
        text_embeddings = pca_text_model(text_embeddings)

    if args.normalize_embedding:
        text_embeddings = normalize_embeddings(text_embeddings)

    predicted_progress_row = []

    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        all_key = list(h5_file[env].keys())
        all_key = [k for k in all_key if "lang" not in k]
        all_trajs = list()

        for key in all_key:
            video_embedding = np.asarray(h5_file[env][key])
            video_embedding = torch.from_numpy(video_embedding).to(device).float()
            if args.subsample_video:
                video_embedding = padding_video(video_embedding, args.max_length)
            if args.pca:
                video_embedding = pca_video_model(video_embedding)

            if args.normalize_embedding:
                traj_data = normalize_embeddings(video_embedding)
            else:
                traj_data = video_embedding

            traj_data = traj_data.unsqueeze(0)
            all_trajs.append(traj_data)
        traj_data = torch.cat(all_trajs, dim=0)
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        video_embedding = video_encoder(traj_data, triangle_mask).squeeze(0)
        last_frame = video_embedding[:, -1, :]

        if args.norm_length:
            last_frame = F.normalize(last_frame, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        total_progress = list()
        for i in range(last_frame.shape[0]):
            use_last_frame = last_frame[i].unsqueeze(0).repeat(text_embeddings.shape[0], 1)
            single_progress = torch.sum(use_last_frame * text_embeddings, dim=1).cpu().detach().numpy() 
            total_progress.append(single_progress)
        total_progress = np.array(total_progress)
        progress = np.mean(total_progress, axis=0)

        
        predicted_progress_row.append(progress)

    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text_list, prob = False)    

