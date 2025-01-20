import h5py
import torch
# from clip_utils import normalize_embeddings, compute_similarity
import json
import random
import numpy as np
# from clip_utils import load_model, embedding_text, embedding_image, SingleLayerMLP
import matplotlib.pyplot as plt
import wandb
import matplotlib
from tqdm import tqdm
import imageio
# from animation_utils import animate_video_with_rewards, log_gif_to_wandb, compute_mmrv, animate_video_with_rewards_class 
import torch.nn.functional as F
from liv import load_liv



matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    model = load_liv()
    model.eval()

    return model


def plot_progress(h5_file, self_attention_model, task_id = None):
    device = next(self_attention_model.parameters()).device

    total_key_name = list(h5_file["text_embeddings"].keys())
    eval_keys = [total_key_name[i] for i in task_id]

    for key in eval_keys:
        text_array = np.asarray(h5_file["text_embeddings"][key])
        text_array = torch.tensor(text_array).to(device).float()

        video_group = h5_file["video_embeddings"][key]
        video_file_list = list(video_group.keys())
        video_file = random.choice(video_file_list)
        video_array = np.asarray(video_group[video_file])
        # reverse the video_array
        video_array = video_array[::-1]
        video_annotations = video_group.attrs["text"]
        video_array = np.array(video_array)
        traj_data = torch.tensor(video_array).to(device).float().squeeze(1)

        predicted_progress = list()

        for i in range(traj_data.shape[0]):
            video_frame_data = traj_data[0:i+1]
            video_frame_data = video_frame_data.unsqueeze(0)

            predicted_score = self_attention_model(video_frame_data, mask=None, text_array=text_array)
            predicted_progress.append(predicted_score.squeeze().detach().cpu().numpy())

        predicted_progress = np.array(predicted_progress)
        frame_index = np.linspace(1, len(predicted_progress), len(predicted_progress))

        if len(task_id) <=2:
            set = "eval"
        else:
            set = "train"
        corr = np.corrcoef(frame_index, predicted_progress)[0, 1]
        wandb.log({f"{set}_corr/{video_annotations}": corr})

        figure = plt.figure()
        plt.plot(frame_index, predicted_progress, label="Correct Text", color="blue")
        plt.xlabel("Frame Index")
        plt.ylabel("Score")
        plt.title(f"{video_annotations}")
        plt.ylim(-1, 1)
        # plt.savefig(f"progress_img/{env}.png")
        plt.legend()
        wandb.log({f"progress_{set}/{video_annotations}": wandb.Image(figure)})
        plt.close()



def plot_matrix_as_image(matrix, names, text):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 2)
    fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    # Plot the matrix with a colormap (darker = higher values)
    cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    # Add color bar
    plt.colorbar(cax)

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(matrix)))
    ax.set_yticks(np.arange(len(matrix)))

    # Label each row and column with the given names
    ax.set_xticklabels(text, rotation=45, ha='left', fontsize=8)
    ax.set_yticklabels(names)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=10)
# keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    wandb.log({f"confusion_matrix/confusion_matrix": wandb.Image(fig)})

    plt.close(fig)  # Close the figure to free memory



def plot_confusion_matrix(h5_file, self_attention_model, train_set):
    device = next(self_attention_model.parameters()).device

    total_text_embedding = list()


    for i in range (8):
        total_key_name = list(h5_file["text_embeddings"].keys())
        eval_keys = total_key_name[i]
        text_array = np.asarray(h5_file["text_embeddings"][eval_keys])
        text_array = torch.tensor(text_array).to(device).float()


        total_text_embedding.append(text_array)

    total_text_embedding = torch.stack(total_text_embedding).to(device).float().squeeze(1)


    predicted_progress_row = []
    total_annotation = []
    for i  in tqdm(range(8)):
        eval_keys = total_key_name[i]
        video_group = h5_file["video_embeddings"][eval_keys]
        video_file_list = list(video_group.keys())
        video_file = random.choice(video_file_list)
        video_array = np.asarray(video_group[video_file])
        video_array = video_array[::-1]
        video_array = np.array(video_array)
        # reverse the video_array
        traj_data = torch.tensor(video_array).to(device).float().squeeze(1)
        # reverse the tensor
        video_annotations = video_group.attrs["text"]
        total_annotation.append(video_annotations)
        # import pdb; pdb.set_trace()
        # traj_data = torch.tensor(video_array).to(device).float().squeeze(1)
        video_frame_data = traj_data[:]

        # reverse the video_frame_data

        video_frame_data = video_frame_data.unsqueeze(0)

        video_frame_data = video_frame_data.repeat(total_text_embedding.shape[0], 1, 1)

        predicted_score = self_attention_model(video_frame_data, mask=None, text_array=total_text_embedding)
        predicted_progress = np.array(predicted_score.squeeze().detach().cpu().numpy())
        predicted_progress_row.append(predicted_progress)
    predicted_progress_row = np.array(predicted_progress_row)
    plot_matrix_as_image(predicted_progress_row, total_annotation, total_annotation)

    # img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text)
    




# def plot_progress_corr(h5_file, model_name, set, self_attention_model, text_pca = None, video_pca = None, transformation_model = None):
#     device = next(self_attention_model.parameters()).device
#     model, processor, tokenizer = load_model(model_name)
#     if set == "train":
#         eval_envs = json.load(open("task_subset.json"))["subset_6"]
#         text = json.load(open("task_subset.json"))["train_annotation"]
#     elif set == "eval":
#         eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
#         text = json.load(open("task_subset.json"))["eval_annotation"]
#     text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
#     text_embeddings = normalize_embeddings(text_embeddings)

#     if text_pca is not None:
#         text_embeddings = text_pca.transform(text_embeddings.detach().cpu().numpy())
#         text_embeddings = torch.tensor(text_embeddings).to(device).float()

#     wandb_dict = {}
#     corrs = []
#     for i  in tqdm(range(len(eval_envs))):
#         env = eval_envs[i]
        
#         model_group = h5_file[model_name]
#         env_group = model_group[env]
#         traj_idx = list(env_group.keys())
#         idx = random.randint(0, len(traj_idx)-1)
#         traj_key = traj_idx[idx]
#         traj_data = np.asarray(env_group[traj_key])
#         env_text_embedding = text_embeddings[i:i+1]
#         # env_text_embedding = env_text_embedding.repeat(traj_data.shape[0], 1)
#         traj_data = torch.tensor(traj_data).to(device).float()
#         traj_data = normalize_embeddings(traj_data)

#         if video_pca is not None:

#             traj_data = video_pca.transform(traj_data.cpu().detach().numpy())
#             traj_data = torch.tensor(traj_data).to(device).float()
#             traj_data = transformation_model(traj_data)

#         diag = list()
#         predicted_progress = list()
        
#         for i in range(traj_data.shape[0]):
#             video_frame_data = traj_data[0:i+1]
#             video_frame_data = video_frame_data.unsqueeze(0)
#             progress_score = self_attention_model(video_frame_data, mask=None, text_array=env_text_embedding)
#             predicted_progress.append(progress_score.squeeze().detach().cpu().numpy())

#         predicted_progress = np.array(predicted_progress)
#         gt_index = np.linspace(1, len(predicted_progress), len(predicted_progress))
#         act_index = np.argsort(predicted_progress) + 1
#         # mmrv = compute_mmrv(gt_index, predicted_progress)
#         # wandb_dict["mmrv/" + set + "/" + env] = mmrv
#         # pearson correlation act_index
#         normed_index = gt_index / len(gt_index)
#         diag_mse = np.mean((normed_index - predicted_progress)**2)

#         diag.append(diag_mse)

#         corr = np.corrcoef(act_index, gt_index)[0, 1]
#         wandb_dict["corr/" + set + "/" + env] = corr
#         wandb_dict["diag_mse/" + set + "/" + env] = diag_mse
#         corrs.append(corr)
#     wandb_dict["mean_corr/" + set] = np.mean(corrs)
#     wandb_dict["diag_mse/" + set] = np.mean(diag)

#     return wandb_dict






# def plot_videos(model_name, self_attention_model, text_pca = None, video_pca = None, transformation_model = None):
#     device = next(self_attention_model.parameters()).device
#     model, processor, tokenizer = load_model(model_name)
#     video_base_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos"
#     video_idxs = ["1", "2"]
#     diffs = ["all_fail", "close_succ", "success", "GT"]
#     tasks = ["button_press_wall", "topdown", "windowclose"]

#     texts = {"button_press_wall": "pressing button from side",
#              "topdown": "pressing button from top",
#              "windowclose": "closing window"}

#     # texts = {"button_press_wall": "Pushing the button from the side",
#     #          "topdown": "Pressing button from top",
#     #          "windowclose": "Closing window"}


#     for task in tasks:
#         text = texts[task]
#         text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
#         text_embeddings = normalize_embeddings(text_embeddings)

#         if text_pca is not None:
#             text_embeddings = text_pca.transform(text_embeddings.detach().cpu().numpy())
#             text_embeddings = torch.tensor(text_embeddings).to(device).float()

#         for diff in diffs:
#             for video_idx in video_idxs:
#                 gif_path = f"{video_base_path}/{task}/{diff}/{video_idx}.gif"
#                 # load gif
#                 frames = imageio.mimread(gif_path)
#                 frames = [frame[:,:,0:3] for frame in frames]

#                 image_embeddings = []
#                 for frame in frames:
#                     image_embedding = embedding_image(model, processor, frame)
#                     image_embedding = normalize_embeddings(image_embedding)
#                     image_embeddings.append(image_embedding)
#                 image_embeddings = torch.stack(image_embeddings).to(device).float().squeeze(1)

#                 if video_pca is not None:
#                     image_embeddings = image_embeddings.view(-1, 1024)
#                     image_embeddings = video_pca.transform(image_embeddings.cpu().detach().numpy())
#                     image_embeddings = torch.tensor(image_embeddings).to(device).float()
#                     image_embeddings = transformation_model(image_embeddings)

#                 predicted_output = list()
#                 for i in range(len(image_embeddings)):
#                     image = image_embeddings[:i+1]

#                     image = image.unsqueeze(0)
#                     progress_score = self_attention_model(image, mask=None, text_array=text_embeddings)
#                     predicted_output.append(progress_score.squeeze().detach().cpu().numpy())
#                 predicted_output = np.array(predicted_output)

#                 frame_index = np.linspace(1, len(predicted_output), len(predicted_output))

#                 figure = plt.figure()
#                 plt.plot(frame_index, predicted_output )
#                 plt.xlabel("Frame Index")
#                 plt.ylabel("Similarity")
#                 plt.title(f"{task} {diff} {video_idx}")
#                 # set y axis range [-1,1]
#                 plt.ylim(-1, 1)

#                 # plt.savefig(f"progress_img/{env}.png")
#                 wandb.log({f"progress_video/{task}/{diff}_{video_idx}": wandb.Image(figure)})
#                 plt.close()
#                 print(f"progress_video/{task}/{diff}/{video_idx}")

                
#                 frames = np.stack(frames)
#                 predicted_output = np.stack(predicted_output)

#                 gt_index = np.linspace(1, len(predicted_output), len(predicted_output))
#                 act_index = np.argsort(predicted_output) + 1

#                 # pearson correlation act_index
#                 corr = np.corrcoef(act_index, gt_index)[0, 1]
#                 wandb.log({f"corr/{task}/{diff}_{video_idx}": corr})



#                 gif_buffer = animate_video_with_rewards(frames, predicted_output, 15)
#                 # mmrv = compute_mmrv(gt_index, predicted_output)
#                 # wandb.log({f"mmrv/{task}/{diff}_{video_idx}": mmrv})
                
#                 log_gif_to_wandb(gif_buffer, f"{task}/{diff}_{video_idx}")


