import h5py
import torch
from clip_utils import normalize_embeddings
import json
import random
import numpy as np
from clip_utils import load_model, embedding_text, embedding_image, SingleLayerMLP
import matplotlib.pyplot as plt
import wandb
import matplotlib
from tqdm import tqdm
import imageio
from animation_utils import animate_video_with_rewards, log_gif_to_wandb, compute_mmrv, animate_video_with_rewards_class
import torch.nn.functional as F
import io
from PIL import Image
from eval_utils import padding_video

matplotlib.use('Agg')

def shorten_name(name, separator=" ", max_length=5):
    parts = name.split(separator)
    return separator.join([part[:max_length] for part in parts])


def plot_matrix_as_image(matrix, names, set, text):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 2)

    fig, ax = plt.subplots(figsize=(len(matrix) * 1.1, len(matrix)))

    # Plot the matrix with a colormap (darker = higher values)
    cax = ax.matshow(
        matrix, cmap="blues", interpolation="nearest"
    )  # originally was viridis

    # Add color bar, make sure it is the same height as the matrix and doesn't overlap with the text
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    shortened_text = [shorten_name(name) for name in text]
    shortened_names = [shorten_name(name) for name in names]

    # Label each row and column with the given names
    ax.set_xticklabels(shortened_text, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(shortened_names)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=10)

    # cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    # plt.colorbar(im, cax=cax)

    # keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    wandb.log({f"confusion_matrix/{set}_confusion_matrix": wandb.Image(fig)})

    # save the figure to disk
    plt.savefig(f"confusion_matrix_{set}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory



def plot_confusion_matrix_pca(h5_file, model_name, set, self_attention_model, pca_text_model = None, pca_video_model = None, transform_model = None, args = None):
    device = next(self_attention_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    if set == "train":
        eval_envs = json.load(open("task_subset.json"))["subset_6"]
        text = json.load(open("task_subset.json"))["train_annotation"]
    elif set == "eval":
        eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
        text = json.load(open("task_subset.json"))["eval_annotation"]
    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
    text_embeddings = normalize_embeddings(text_embeddings)
    if pca_text_model is not None:
        text_embeddings = pca_text_model.transform(text_embeddings.cpu().detach().numpy())
        text_embeddings = torch.tensor(text_embeddings).to(device).float()


    predicted_progress_row = []
    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        
        model_group = h5_file[model_name]
        env_group = model_group[env]
        traj_idx = list(env_group.keys())
        idx = random.randint(0, len(traj_idx)-1)
        traj_key = traj_idx[idx]
        traj_data = np.asarray(env_group[traj_key])

        traj_data = torch.tensor(traj_data).to(device).float()
        if args.subsample_video:
            traj_data = padding_video(traj_data, args.max_length)

        if args.normalize_embedding:
            traj_data = normalize_embeddings(traj_data)
        if pca_video_model is not None:
            traj_data = traj_data.view(-1, 1024)
            traj_data = pca_video_model.transform(traj_data.cpu().detach().numpy())
            traj_data = torch.tensor(traj_data).to(device).float()
            traj_data = transform_model(traj_data)


        # for i in range(traj_data.shape[0]):

        video_frame_data = traj_data[:]
        video_frame_data = video_frame_data.unsqueeze(0)
        video_frame_data = video_frame_data.repeat(text_embeddings.shape[0], 1, 1)

        predicted_score = self_attention_model(video_frame_data, mask=None, text_array=text_embeddings)
        predicted_progress = np.array(predicted_score.squeeze().detach().cpu().numpy())
        predicted_progress_row.append(predicted_progress)
    predicted_progress_row = np.array(predicted_progress_row)

    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text)



def plot_confusion_matrix_pca_class(h5_file, model_name, set, self_attention_model, pca_text_model = None, pca_video_model = None, transform_model = None, args = None):
    device = next(self_attention_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    if set == "train":
        eval_envs = json.load(open("task_subset.json"))["subset_6"]
        text = json.load(open("task_subset.json"))["train_annotation"]
    elif set == "eval":
        eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
        text = json.load(open("task_subset.json"))["eval_annotation"]
    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
    if args.normalize_embedding:
        text_embeddings = normalize_embeddings(text_embeddings)
    if pca_text_model is not None:
        text_embeddings = pca_text_model.transform(text_embeddings.cpu().detach().numpy())
        text_embeddings = torch.tensor(text_embeddings).to(device).float()


    predicted_progress_row = []
    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        
        model_group = h5_file[model_name]
        env_group = model_group[env]
        traj_idx = list(env_group.keys())
        idx = random.randint(0, len(traj_idx)-1)
        traj_key = traj_idx[idx]
        traj_data = np.asarray(env_group[traj_key])

        traj_data = torch.tensor(traj_data).to(device).float()
        if args.subsample_video:
            traj_data = padding_video(traj_data, args.max_length)
        if args.normalize_embedding:
            traj_data = normalize_embeddings(traj_data)
        if pca_video_model is not None:
            traj_data = traj_data.view(-1, 1024)
            traj_data = pca_video_model.transform(traj_data.cpu().detach().numpy())
            traj_data = torch.tensor(traj_data).to(device).float()
            traj_data = transform_model(traj_data)


        # for i in range(traj_data.shape[0]):

        video_frame_data = traj_data[:]
        video_frame_data = video_frame_data.unsqueeze(0)
        video_frame_data = video_frame_data.repeat(text_embeddings.shape[0], 1, 1)

        pred_class = self_attention_model(video_frame_data, mask=None, text_array=text_embeddings)
        class_label = torch.argmax(pred_class, dim = 1)
        # non_zero_index = class_label != 0
        # predicted_score[non_zero_index] = 0


        predicted_progress = np.array(class_label.squeeze().detach().cpu().numpy())
        predicted_progress_row.append(predicted_progress)
    predicted_progress_row = np.array(predicted_progress_row)

    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text)
    
