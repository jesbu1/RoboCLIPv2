import h5py
import torch
from clip_utils import normalize_embeddings, compute_similarity
import json
import random
import numpy as np
from clip_utils import load_model, embedding_text, embedding_image, SingleLayerMLP
import matplotlib.pyplot as plt
import wandb
import matplotlib
from tqdm import tqdm
import imageio
from animation_utils import animate_video_with_rewards, log_gif_to_wandb, compute_mmrv
import torch.nn.functional as F


matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_progress(h5_file, model_name, transform_model, set, self_attention_model):
    device = next(transform_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    if set == "train":
        eval_envs = json.load(open("task_subset.json"))["subset_6"]
        text = json.load(open("task_subset.json"))["train_annotation"]
    elif set == "eval":
        eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
        text = json.load(open("task_subset.json"))["eval_annotation"]
    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
    text_embeddings = normalize_embeddings(text_embeddings)


    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        
        model_group = h5_file[model_name]
        env_group = model_group[env]
        traj_idx = list(env_group.keys())
        idx = random.randint(0, len(traj_idx)-1)
        traj_key = traj_idx[idx]
        traj_data = np.asarray(env_group[traj_key])
        env_text_embedding = text_embeddings[i:i+1]
        # env_text_embedding = env_text_embedding.repeat(traj_data.shape[0], 1)
        traj_data = torch.tensor(traj_data).to(device).float()
        traj_data = normalize_embeddings(traj_data)

        weighted_embeddings = list()
        for i in range(traj_data.shape[0]):
            video_frame_data = traj_data[0:i+1]
            video_frame_data = video_frame_data.unsqueeze(0)

            video_embedding = self_attention_model(video_frame_data)
            progress_input = video_embedding - env_text_embedding

            weighted_embeddings.append(progress_input)
        weighted_embeddings = torch.cat(weighted_embeddings, dim=0)
        predicted_progress = transform_model(weighted_embeddings).squeeze().detach().cpu().numpy()

        frame_index = np.linspace(1, len(predicted_progress), len(predicted_progress))

        figure = plt.figure()
        plt.plot(frame_index, predicted_progress)
        plt.xlabel("Frame Index")
        plt.ylabel("Score")
        plt.title(f"{env}")
        plt.ylim(-1, 1)
        # plt.savefig(f"progress_img/{env}.png")
        wandb.log({f"progress_{set}/{env}": wandb.Image(figure)})
        plt.close()






def plot_progress_corr(h5_file, model_name, transform_model, set, self_attention_model):
    device = next(transform_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    if set == "train":
        eval_envs = json.load(open("task_subset.json"))["subset_6"]
        text = json.load(open("task_subset.json"))["train_annotation"]
    elif set == "eval":
        eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
        text = json.load(open("task_subset.json"))["eval_annotation"]
    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
    text_embeddings = normalize_embeddings(text_embeddings)



    wandb_dict = {}
    corrs = []
    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        
        model_group = h5_file[model_name]
        env_group = model_group[env]
        traj_idx = list(env_group.keys())
        idx = random.randint(0, len(traj_idx)-1)
        traj_key = traj_idx[idx]
        traj_data = np.asarray(env_group[traj_key])
        env_text_embedding = text_embeddings[i:i+1]
        env_text_embedding = env_text_embedding.repeat(traj_data.shape[0], 1)
        traj_data = torch.tensor(traj_data).to(device).float()
        traj_data = normalize_embeddings(traj_data)

        mean_embeddings = list()
        diag = []
        for i in range(traj_data.shape[0]):
            video_frame_data = traj_data[0:i+1]
            video_frame_data = video_frame_data.unsqueeze(0)

            video_embedding = self_attention_model(video_frame_data)
            progress_input = video_embedding - env_text_embedding
            mean_embeddings.append(progress_input)

        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        predicted_progress = transform_model(mean_embeddings).squeeze().detach().cpu().numpy()
                
        gt_index = np.linspace(1, len(predicted_progress), len(predicted_progress))
        act_index = np.argsort(predicted_progress) + 1
        # mmrv = compute_mmrv(gt_index, predicted_progress)
        # wandb_dict["mmrv/" + set + "/" + env] = mmrv
        # pearson correlation act_index
        normed_index = gt_index / len(gt_index)
        diag_mse = np.mean((normed_index - predicted_progress)**2)

        diag.append(diag_mse)

        corr = np.corrcoef(act_index, gt_index)[0, 1]
        wandb_dict["corr/" + set + "/" + env] = corr
        wandb_dict["diag_mse/" + set + "/" + env] = diag_mse
        corrs.append(corr)
    wandb_dict["mean_corr/" + set] = np.mean(corrs)
    wandb_dict["diag_mse/" + set] = np.mean(diag)

    return wandb_dict






def plot_videos(model_name, transform_model, self_attention_model):
    device = next(transform_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    video_base_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos"
    video_idxs = ["1", "2"]
    diffs = ["all_fail", "close_succ", "success", "GT"]
    tasks = ["button_press_wall", "topdown", "windowclose"]

    texts = {"button_press_wall": "Robot pressing button from side",
             "topdown": "Robot pressing button from top",
             "windowclose": "Robot closing window"}

    # texts = {"button_press_wall": "Pushing the button from the side",
    #          "topdown": "Pressing button from top",
    #          "windowclose": "Closing window"}


    for task in tasks:
        text = texts[task]
        text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
        text_embeddings = normalize_embeddings(text_embeddings)


        for diff in diffs:
            for video_idx in video_idxs:
                gif_path = f"{video_base_path}/{task}/{diff}/{video_idx}.gif"
                # load gif
                frames = imageio.mimread(gif_path)
                frames = [frame[:,:,0:3] for frame in frames]

                image_embeddings = []
                for frame in frames:
                    image_embedding = embedding_image(model, processor, frame)
                    image_embedding = normalize_embeddings(image_embedding)
                    image_embeddings.append(image_embedding)
                image_embeddings = torch.stack(image_embeddings).to(device).float().squeeze(1)

                predicted_output = list()
                for i in range(len(image_embeddings)):
                    image = image_embeddings[:i+1]

                    image = image.unsqueeze(0)
                    video_embedding = self_attention_model(image)
                    progress_input = video_embedding - text_embeddings

                    predicted_progress = transform_model(progress_input).squeeze().detach().cpu().numpy()
                    predicted_output.append(predicted_progress)

                frame_index = np.linspace(1, len(predicted_output), len(predicted_output))

                figure = plt.figure()
                plt.plot(frame_index, predicted_output )
                plt.xlabel("Frame Index")
                plt.ylabel("Similarity")
                plt.title(f"{task} {diff} {video_idx}")
                # set y axis range [-1,1]
                plt.ylim(-1, 1)

                # plt.savefig(f"progress_img/{env}.png")
                wandb.log({f"progress_video/{task}/{diff}_{video_idx}": wandb.Image(figure)})
                plt.close()
                print(f"progress_video/{task}/{diff}/{video_idx}")

                
                frames = np.stack(frames)
                predicted_output = np.stack(predicted_output)

                gt_index = np.linspace(1, len(predicted_output), len(predicted_output))
                act_index = np.argsort(predicted_output) + 1

                # pearson correlation act_index
                corr = np.corrcoef(act_index, gt_index)[0, 1]
                wandb.log({f"corr/{task}/{diff}_{video_idx}": corr})



                gif_buffer = animate_video_with_rewards(frames, predicted_output, 15)
                mmrv = compute_mmrv(gt_index, predicted_output)
                wandb.log({f"mmrv/{task}/{diff}_{video_idx}": mmrv})
                
                log_gif_to_wandb(gif_buffer, f"{task}/{diff}_{video_idx}")



