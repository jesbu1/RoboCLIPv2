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
from new_task_annotation_v2 import train_gt_annotation, eval_gt_annotation
import textwrap

matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_subset = json.load(open("new_task_v2.json"))
train_tasks = task_subset["training_tasks"]
eval_tasks = task_subset["eval_tasks"]




def shorten_name(name, separator=" ", max_length=5):
    parts = name.split(separator)
    return separator.join([part[:max_length] for part in parts])
    
def shorten_text(text, max_length=25):
    print(f"og_text: {text}")
    shortened_text = textwrap.shorten(text, width=max_length, placeholder="...")
    print(f"shortened_text: {shortened_text}")
    return shortened_text

def plot_matrix_as_image(matrix, names, set, text):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 1)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.25, len(matrix) * 1))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)

    # 只保留两位小数
    cbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
    cbar.update_ticks()  # 更新刻度标签

    # 放大颜色条字体
    cbar.ax.yaxis.set_tick_params(labelsize=16)  # 你可以调整 `fontsize`

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    # shortened_text = [shorten_name(name, max_length = 6) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 6) for name in names]

    shortened_text = [shorten_text(name, max_length = 25) for name in text]
    shortened_names = [shorten_name(name, separator = "-", max_length = 12) for name in names]

    # Label each row and column with the given names
    ax.set_xticklabels(shortened_text, rotation=30, ha='left', fontsize=18)
    ax.set_yticklabels(shortened_names, fontsize=18)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=20)
    # keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    wandb.log({f"confusion_matrix/{set}_confusion_matrix": wandb.Image(fig)})
    plt.savefig(f"confusion_matrix_{set}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory





def plot_confusion_matrix_pca_class(h5_file, set, self_attention_model, args):
    device = next(self_attention_model.parameters()).device
    if set == "train":
        eval_envs = train_tasks
        env_anns = train_gt_annotation
    else:
        eval_envs = eval_tasks
        env_anns = eval_gt_annotation
    embedding_list = list()
    text = []
    for env in eval_envs:
        env_name = env + "_text"
        embedding_list.append(np.asarray(h5_file[env_name]))
        text.append(env_anns[env])
    embedding_list = np.stack(embedding_list)
    text_embeddings = torch.tensor(embedding_list).to(device).float()
    
    if args.normalize_embedding:
        text_embeddings = normalize_embeddings(text_embeddings)

    predicted_progress_row = []
    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        traj_data = np.asarray(h5_file[env])

        traj_data = torch.tensor(traj_data).to(device).float()
        if args.subsample_video:
            traj_data = padding_video(traj_data, args.max_length)
        if args.normalize_embedding:
            traj_data = normalize_embeddings(traj_data)

        
        traj_data = traj_data.view(-1, 1024).unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1)
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        mask = torch.ones(traj_data.shape[1]).to(device).unsqueeze(0).repeat(traj_data.shape[0], 1)

        pred_class, two_step_class = self_attention_model(traj_data, triangle_mask, text_embeddings, mask)
        
        batch_size, seq_len, _ = traj_data.size()
        if args.catagorical_progress:
            pred_class = torch.argmax(pred_class, dim = 1)
        else:
            pred_class = pred_class.squeeze(1)


        if args.two_step_training:
            pred_two_class = torch.argmax(two_step_class, dim = 1)
            pred_class = pred_two_class * pred_class

        pred_class = pred_class.view(batch_size, seq_len)


        predicted_progress = np.array(pred_class.squeeze().detach().cpu().numpy())
        predicted_progress_row.append(predicted_progress[:,-1])

    predicted_progress_row = np.array(predicted_progress_row)

    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text)
    
def plot_confusion_matrix_pca_class_pdf(h5_file, model_name, set, self_attention_model, args):
    device = next(self_attention_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    if set == "train":
        eval_envs = json.load(open("task_subset.json"))["subset_6"]
        text = json.load(open("task_subset.json"))["train_annotation"]
        display_text = json.load(open("task_subset.json"))["train_dis_annotation"]
    elif set == "eval":
        eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
        text = json.load(open("task_subset.json"))["eval_annotation"]
        display_text = json.load(open("task_subset.json"))["eval_dis_annotation"]

    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
    if args.normalize_embedding:
        text_embeddings = normalize_embeddings(text_embeddings)

    predicted_progress_row = []
    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        

        traj_data = np.asarray(h5_file[env])

        traj_data = torch.tensor(traj_data).to(device).float()
        if args.subsample_video:
            traj_data = padding_video(traj_data, args.max_length)
        if args.normalize_embedding:
            traj_data = normalize_embeddings(traj_data)

        
        traj_data = traj_data.view(-1, 1024).unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1)
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        mask = torch.ones(traj_data.shape[1]).to(device).unsqueeze(0).repeat(traj_data.shape[0], 1)

        pred_class, two_step_class = self_attention_model(traj_data, triangle_mask, text_embeddings, mask)
        
        batch_size, seq_len, _ = traj_data.size()
        if args.catagorical_progress:
            pred_class = torch.argmax(pred_class, dim = 1)
        else:
            pred_class = pred_class.squeeze(1)


        if args.two_step_training:
            pred_two_class = torch.argmax(two_step_class, dim = 1)
            pred_class = pred_two_class * pred_class

        pred_class = pred_class.view(batch_size, seq_len)
        pred_class = torch.clamp(pred_class, 0, 1)


        predicted_progress = np.array(pred_class.squeeze().detach().cpu().numpy())
        predicted_progress_row.append(predicted_progress[:,-1])

    predicted_progress_row = np.array(predicted_progress_row)
    eval_envs = [eval_envs[i].split("-v")[0] for i in range(len(eval_envs))]
    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, display_text)

    

