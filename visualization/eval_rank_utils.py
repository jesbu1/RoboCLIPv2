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

matplotlib.use('Agg')
def plot_progress_eval(h5_file, model_name, transform_model):
    device = next(transform_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
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
        env_text_embedding = env_text_embedding.repeat(traj_data.shape[0], 1)
        traj_data = torch.tensor(traj_data).to(device).float()

        input_embedding = torch.cat([env_text_embedding, traj_data], dim=1)
        output_embedding = transform_model(input_embedding)
        
        similarity = compute_similarity(env_text_embedding, output_embedding)
        
        frame_index = np.linspace(1, len(similarity), len(similarity))

        figure = plt.figure()
        plt.plot(frame_index, similarity.detach().cpu().numpy())
        plt.xlabel("Frame Index")
        plt.ylabel("Similarity")
        plt.title(f"{env}")
        # plt.savefig(f"progress_img/{env}.png")
        wandb.log({f"progress_eval/{env}": wandb.Image(figure)})
        plt.close()

def plot_progress_train(h5_file, model_name, transform_model):
    device = next(transform_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    eval_envs = json.load(open("task_subset.json"))["subset_6"]
    text = json.load(open("task_subset.json"))["train_annotation"]
    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()


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

        input_embedding = torch.cat([env_text_embedding, traj_data], dim=1)
        output_embedding = transform_model(input_embedding)
        
        similarity = compute_similarity(env_text_embedding, output_embedding)
        
        frame_index = np.linspace(1, len(similarity), len(similarity))

        figure = plt.figure()
        plt.plot(frame_index, similarity.detach().cpu().numpy())
        plt.xlabel("Frame Index")
        plt.ylabel("Similarity")
        plt.title(f"{env}")
        # plt.savefig(f"progress_img/{env}.png")
        wandb.log({f"progress_train/{env}": wandb.Image(figure)})
        plt.close()

    

def plot_progress_corr(h5_file, model_name, transform_model, set, text_pca_model, image_pca_model, linear_model, subtract=False):
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
    if text_pca_model is not None:
        text_embeddings = text_pca_model.transform(text_embeddings.detach().cpu().numpy())
        text_embeddings = torch.tensor(text_embeddings).to(device).float()

    wandb_dict = {}
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
        if image_pca_model is not None:
            traj_data = image_pca_model.transform(traj_data.detach().cpu().numpy())
            traj_data = torch.tensor(traj_data).to(device).float()
            traj_data = linear_model(traj_data)

        if subtract:
            input_embedding = traj_data - env_text_embedding
        else:
            input_embedding = torch.cat([env_text_embedding, traj_data], dim=1)
        predicted_progress = transform_model(input_embedding).squeeze().detach().cpu().numpy()
                
        gt_index = np.linspace(1, len(predicted_progress), len(predicted_progress))
        act_index = np.argsort(predicted_progress) + 1

        # pearson correlation act_index
        corr = np.corrcoef(act_index, gt_index)[0, 1]
        wandb_dict["corr/" + set + "/" + env] = corr

    return wandb_dict


def plot_progress(h5_file, model_name, transform_model, set, text_pca_model, image_pca_model, linear_model, subtract=False):
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

    if text_pca_model is not None:
        text_embeddings = text_pca_model.transform(text_embeddings.detach().cpu().numpy())
        text_embeddings = torch.tensor(text_embeddings).to(device).float()

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

        if image_pca_model is not None:
            traj_data = image_pca_model.transform(traj_data.detach().cpu().numpy())
            traj_data = torch.tensor(traj_data).to(device).float()
            traj_data = linear_model(traj_data)
        
        if subtract:
            input_embedding = traj_data - env_text_embedding
        else:
            input_embedding = torch.cat([env_text_embedding, traj_data], dim=1)
        predicted_progress = transform_model(input_embedding).squeeze().detach().cpu().numpy()
                
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
    

def plot_videos(model_name, transform_model, text_pca_model, image_pca_model, linear_model, subtract=False):
    device = next(transform_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    video_base_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos"
    video_idxs = ["1", "2"]
    diffs = ["all_fail", "close_succ", "success", "GT"]
    tasks = ["button_press_wall", "topdown", "windowclose"]

    texts = {"button_press_wall": "Robot pressing button from side",
             "topdown": "Robot pressing button from top",
             "windowclose": "Robot closing window"}
    

    for task in tasks:
        text = texts[task]
        text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
        text_embeddings = normalize_embeddings(text_embeddings)

        if text_pca_model is not None:
            text_embeddings = text_pca_model.transform(text_embeddings.detach().cpu().numpy())
            text_embeddings = torch.tensor(text_embeddings).to(device).float()

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

                if image_pca_model is not None:
                    image_embeddings = image_pca_model.transform(image_embeddings.detach().cpu().numpy())
                    image_embeddings = torch.tensor(image_embeddings).to(device).float()
                    image_embeddings = linear_model(image_embeddings)

                env_text_embedding = text_embeddings.repeat(image_embeddings.shape[0], 1)
                if subtract:
                    input_embedding = image_embeddings - env_text_embedding
                else:
                    input_embedding = torch.cat([env_text_embedding, image_embeddings], dim=1)
                predicted_progress = transform_model(input_embedding).squeeze().detach().cpu().numpy()

                frame_index = np.linspace(1, len(predicted_progress), len(predicted_progress))

                figure = plt.figure()
                plt.plot(frame_index, predicted_progress)
                plt.xlabel("Frame Index")
                plt.ylabel("Similarity")
                plt.title(f"{task} {diff} {video_idx}")
                # set y axis range [-1,1]
                plt.ylim(-1, 1)

                # plt.savefig(f"progress_img/{env}.png")
                wandb.log({f"progress_video/{task}/{diff}_{video_idx}": wandb.Image(figure)})
                plt.close()
                print(f"progress_video/{task}/{diff}/{video_idx}")







if __name__ == "__main__":

    h5_file = h5py.File("/scr/jzhang96/metaworld_25_for_clip_liv.h5", "r")
    model_name = "clip"
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = SingleLayerMLP(768 + 768, 768)
    model = model.to(device)
    plot_progress(h5_file, model_name, model)
    









