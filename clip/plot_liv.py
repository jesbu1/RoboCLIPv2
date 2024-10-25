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
from animation_utils import animate_video_with_rewards, log_gif_to_wandb

matplotlib.use('Agg')

# def plot_progress(h5_file, model_name, transform_model, set, subtract=False):
#     device = next(transform_model.parameters()).device
#     model, processor, tokenizer = load_model(model_name)
#     if set == "train":
#         eval_envs = json.load(open("task_subset.json"))["subset_6"]
#         text = json.load(open("task_subset.json"))["train_annotation"]
#     elif set == "eval":
#         eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
#         text = json.load(open("task_subset.json"))["eval_annotation"]
#     text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
#     text_embeddings = normalize_embeddings(text_embeddings)


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

#         mean_embeddings = list()
#         for i in range(traj_data.shape[0]):
#             mean_data = traj_data[0:i+1]
#             mean_data = torch.mean(mean_data, dim=0).unsqueeze(0)
#             if subtract:
#                 progress_input = mean_data - env_text_embedding
#             else:
#                 progress_input = torch.cat([env_text_embedding, mean_data], dim=1)
#             mean_embeddings.append(progress_input)
#         mean_embeddings = torch.cat(mean_embeddings, dim=0)
#         predicted_progress = transform_model(mean_embeddings).squeeze().detach().cpu().numpy()


#         frame_index = np.linspace(1, len(predicted_progress), len(predicted_progress))

#         figure = plt.figure()
#         plt.plot(frame_index, predicted_progress)
#         plt.xlabel("Frame Index")
#         plt.ylabel("Score")
#         plt.title(f"{env}")
#         plt.ylim(-1, 1)
#         # plt.savefig(f"progress_img/{env}.png")
#         wandb.log({f"progress_{set}/{env}": wandb.Image(figure)})
#         plt.close()


def plot_progress_corr(h5_file, model_name, set, subtract=False):
    # device = next(transform_model.parameters()).device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        traj_data = torch.tensor(traj_data).to(device).float()
        traj_data = normalize_embeddings(traj_data)

        simi_score = list()
        for j in range(traj_data.shape[0]):
            frame_embedding = traj_data[j:j+1]
            cos_sim = model.module.sim(frame_embedding, env_text_embedding).item()
            simi_score.append(cos_sim)

        gt_index = np.linspace(1, len(simi_score), len(simi_score))
        act_index = np.argsort(simi_score) + 1

        # pearson correlation act_index
        corr = np.corrcoef(act_index, gt_index)[0, 1]
        wandb_dict["corr/" + set + "/" + env] = corr
        corrs.append(corr)
    wandb_dict["mean_corr/" + set] = np.mean(corrs)

    return wandb_dict



def plot_videos(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

                simi_score = list()
                for i in range(len(image_embeddings)):
                    image = image_embeddings[i:i+1]
                    cos_sim = model.module.sim(image, text_embeddings).item()
                    simi_score.append(cos_sim)


                frame_index = np.linspace(1, len(simi_score), len(simi_score))

                figure = plt.figure()
                plt.plot(frame_index, simi_score )
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
                predicted_output = np.stack(simi_score)
                gif_buffer = animate_video_with_rewards(frames, predicted_output, 15)
                
                log_gif_to_wandb(gif_buffer, f"{task}/{diff}_{video_idx}")


if __name__ == "__main__":
    h5_file = h5py.File('/scr/jzhang96/metaworld_25_for_clip_liv.h5', "r")
    model_name = "liv"

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "LivZeroShot" 

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="LivZeroShot",
        name=experiment_name,
    )

    train_dict = plot_progress_corr(h5_file, model_name, "train")
    eval_dict = plot_progress_corr(h5_file, model_name, "eval")
    wandb.log(train_dict)
    wandb.log(eval_dict)
    plot_videos(model_name)

    