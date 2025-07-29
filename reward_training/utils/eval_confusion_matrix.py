import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


def plot_matrix_as_image(matrix, names, set, text, prob = False, org_progress = False, epoch = None, ema = False):
    # Create a figure and axis
    # only keep 2 decimal points

    diagonal = np.diag(matrix)
    diagonal_sum = np.sum(diagonal)
    total_sum = np.sum(matrix)
    rest_sum = total_sum - diagonal_sum
    score = rest_sum / diagonal_sum
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
        if ema:
            wandb.log({f"EMA_confusion_matrix_same_class_prob/{set}_prob_confusion_matrix": wandb.Image(fig, caption=f"Epoch {epoch}")})
        else:
            wandb.log({f"confusion_matrix_same_class_prob/{set}_prob_confusion_matrix": wandb.Image(fig, caption=f"Epoch {epoch}")})
    elif org_progress:
        if ema:
            wandb.log({f"EMA_confusion_matrix_org_progress_no_two_step/{set}_original_progress_confusion_matrix": wandb.Image(fig, caption=f"Epoch {epoch}_Score: {score:.2f}")})
        else:
            wandb.log({f"confusion_matrix_org_progress_no_two_step/{set}_original_progress_confusion_matrix": wandb.Image(fig, caption=f"Epoch {epoch}_Score: {score:.2f}")})
    else:
        if ema:
            wandb.log({f"EMA_confusion_matrix/{set}_confusion_matrix": wandb.Image(fig, caption=f"Epoch {epoch}_Score: {score:.2f}")})
        else:
            wandb.log({f"confusion_matrix/{set}_confusion_matrix": wandb.Image(fig, caption=f"Epoch {epoch}_Score: {score:.2f}")})
    # plt.savefig(f"confusion_matrix_{set}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory


def plot_matrix_as_image_for_paper(matrix, names, set, text, epoch = None, run_name = None):
    # Create a figure and axis
    # only keep 2 decimal points

    matrix = np.array(matrix)
    m_min = matrix.min()
    m_max = matrix.max()

    if m_max == m_min:
        # 说明整张矩阵所有值相同，可以直接都置为0 或 1
        # 这里演示直接设置为 0
        matrix= np.zeros_like(matrix)
    else:
        matrix = (matrix - m_min) / (m_max - m_min)

    # 只保留两位小数
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.25, len(matrix) * 1))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    # cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)

    # # 只保留两位小数
    # cbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
    # cbar.update_ticks()  # 更新刻度标签

    # # 放大颜色条字体
    # cbar.ax.yaxis.set_tick_params(labelsize=16)  # 你可以调整 `fontsize`

    # Set x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # shortened_text = [shorten_name(name, max_length = 6) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 6) for name in names]

    # shortened_text = [shorten_text(name, max_length = 25) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 12) for name in names]

    # Label each row and column with the given names
    # ax.set_xticklabels(shortened_text, rotation=30, ha='left', fontsize=18)
    # ax.set_yticklabels(shortened_names, fontsize=18)

    # Display the values in the matrix
    # for (i, j), val in np.ndenumerate(matrix):
    #     ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=20)

    # keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    wandb.log({f"confusion_matrix_for_paper/{set}_confusion_matrix_Rewind": wandb.Image(fig, caption=f"Epoch {epoch}")})
    folder_name = run_name
    if not os.path.exists(f"confusion_matrix_for_paper"):
        os.makedirs(f"confusion_matrix_for_paper")
    if not os.path.exists(f"confusion_matrix_for_paper/{folder_name}"):
        os.makedirs(f"confusion_matrix_for_paper/{folder_name}")
    pdf_path = f"confusion_matrix_for_paper/{folder_name}/confusion_matrix_{set}_epoch_{epoch}.pdf"
                          
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory





def plot_confusion_matrix(h5_file, set, self_attention_model, args, epoch = None, ema = False, run_name = None):
    device = next(self_attention_model.parameters()).device

    keys = list(h5_file.keys())
    eval_envs = keys


    text_embeddings = []
    text_list = []
    for key in eval_envs:
        embedding = np.asarray(h5_file[key]["minilm_lang_embedding"])[0].reshape(1, -1)

        text_embeddings.append(embedding)
        text_list.append(key)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    text_embeddings = torch.from_numpy(text_embeddings).to(device).float()


    pred_org_progress_list = []

    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        choose_keys = list(h5_file[env].keys())
        choose_keys = [key for key in choose_keys if "lang" not in key]

        traj_list = []

        for key in choose_keys:
            video_embedding = np.asarray(h5_file[env][key])
            if args.subsample_video:
                video_embedding = padding_video(video_embedding, args.max_length)
            traj_list.append(video_embedding)
        traj_data_all = np.stack(traj_list, axis=0)
        traj_data_all = torch.from_numpy(traj_data_all).to(device).float()


        progress_result_list = []
        progress_prob_list = []
        progress_org_list = []
        for id in range(traj_data_all.shape[0]):
            traj_data = traj_data_all[id].unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1)
            pred_class = self_attention_model(traj_data, text_embeddings)
            
            pred_class = pred_class[:, -1].squeeze()
            progress_org_list.append(pred_class.clone().cpu().detach().numpy())


        progress_org_list = np.stack(progress_org_list, axis=0)
        progress_org_list = np.mean(progress_org_list, axis=0)
        pred_org_progress_list.append(progress_org_list)


    pred_matrix = np.array(pred_org_progress_list)

    img2 = plot_matrix_as_image(pred_org_progress_list, eval_envs, set, text_list, prob = False, org_progress = True, epoch = epoch, ema = ema)
    img3 = plot_matrix_as_image_for_paper(pred_org_progress_list, eval_envs, set, text_list, epoch = epoch, run_name = run_name)





def plot_confusion_matrix_token(h5_file, set, self_attention_model, args):
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
    text_seq_list = []
    for key in eval_envs:
        
        text_embedding = np.asarray(h5_file[key]["lang_embedding_individual"])
        text_embedding = torch.tensor(text_embedding).to(device).float()
        bs, seq_len, _ = text_embedding.size()

        text_embeddings.append(text_embedding)
        text_list.append(key)
        text_seq_list.append(seq_len)
    # text_embeddings = torch.tensor(text_embeddings).to(device).float()
    

    predicted_progress_row = []

    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        video_embedding = np.asarray(h5_file[env]["1"])
        
        video_embedding = torch.tensor(video_embedding).to(device).float()
        if args.subsample_video:
            video_embedding = padding_video(video_embedding, args.max_length)

        traj_data = video_embedding
        traj_data = traj_data.unsqueeze(0)
        
        text_progress_row = []
        text_prob_row = []
        for j in range(len(text_embeddings)):
            input_feature = torch.cat([text_embeddings[j], traj_data], dim=1)
            text_len = text_seq_list[j]
            feature_len = input_feature.size(1)

            triangle_mask = torch.tril(torch.ones(feature_len, feature_len)).to(device).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
            video_mask = torch.ones(1, feature_len).to(device)
            video_mask[:, :text_len] = 0
            progress_output, _ = self_attention_model(input_feature, triangle_mask, text_len, mask = video_mask)
            pred_class = progress_output.view(-1)
            
            batch_size, seq_len, _ = traj_data.size()
            pred_class = pred_class.squeeze()
            text_progress_row.append(pred_class[-1])
        text_progress_row = torch.stack(text_progress_row).detach().cpu().numpy()


        predicted_progress_row.append(text_progress_row)

    predicted_progress_row = np.array(predicted_progress_row)
    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text_list, prob = False)
    

    
