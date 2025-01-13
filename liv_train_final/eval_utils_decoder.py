import h5py
import torch
from clip_utils import normalize_embeddings
import json
import random
import numpy as np
from clip_utils import load_model, embedding_text, embedding_image
import matplotlib.pyplot as plt
import wandb
import matplotlib
from tqdm import tqdm
import imageio
from animation_utils import animate_video_with_rewards, log_gif_to_wandb, compute_mmrv, animate_video_with_rewards_class
import torch.nn.functional as F
import copy


matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


def plot_progress(h5_file, model_name, set, self_attention_model, args):
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

    wrong_text = "Hello World"
    wrong_text_embedding = embedding_text(model, tokenizer, [wrong_text]).to(device).float()
    if args.normalize_embedding:
        wrong_text_embedding = normalize_embeddings(wrong_text_embedding)


    wandb_dict = {}
    corrs = list()
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

        if args.subsample_video:
            traj_data = padding_video(traj_data, args.max_length)

        if args.normalize_embedding:
            traj_data = normalize_embeddings(traj_data)

        predicted_progress = list()
        wrong_text_progress = list()
        diag = list()
        
        for i in range(traj_data.shape[0]):
            video_frame_data = traj_data[0:i+1]
            video_frame_data = video_frame_data.unsqueeze(0)

            predicted_score, two_step_class = self_attention_model(video_frame_data, mask=None, text_array=env_text_embedding)
            if args.two_step_training:
                pred_two_class = torch.argmax(two_step_class, dim = 1)
                predicted_score = predicted_score * pred_two_class.unsqueeze(1)
            predicted_progress.append(predicted_score.squeeze().detach().cpu().numpy())
            wrong_text_score, wrong_two_step_class = self_attention_model(video_frame_data, mask=None, text_array=wrong_text_embedding)
            if args.two_step_training:
                wrong_pred_two_class = torch.argmax(wrong_two_step_class, dim = 1)
                wrong_text_score = wrong_text_score * wrong_pred_two_class.unsqueeze(1)
            wrong_text_progress.append(wrong_text_score.squeeze().detach().cpu().numpy())
        predicted_progress = np.array(predicted_progress)
        wrong_text_progress = np.array(wrong_text_progress)
        frame_index = np.linspace(1, len(predicted_progress), len(predicted_progress))
        act_index = np.argsort(predicted_progress) + 1
        normed_index = frame_index / len(frame_index)
        diag_mse = np.mean((normed_index - predicted_progress)**2)
        corr = np.corrcoef(act_index, frame_index)[0, 1]

        diag.append(diag_mse)
        wandb_dict["corr/" + set + "/" + env] = corr
        wandb_dict["diag_mse/" + set + "/" + env] = diag_mse
        corrs.append(corr)

        figure = plt.figure()
        plt.plot(frame_index, predicted_progress, label="Correct Text", color="blue")
        plt.plot(frame_index, wrong_text_progress, label="Hello World", color="red")
        plt.xlabel("Frame Index")
        plt.ylabel("Score")
        plt.title(f"{env}")
        plt.ylim(-1, 1)
        # plt.savefig(f"progress_img/{env}.png")
        plt.legend()
        wandb.log({f"progress_{set}/{env}": wandb.Image(figure)})
        plt.close()
    wandb_dict["mean_corr/" + set] = np.mean(corrs)
    wandb_dict["diag_mse/" + set] = np.mean(diag)
    wandb.log(wandb_dict)




def plot_progress_class(h5_file, model_name, set, self_attention_model, args):
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

    wrong_text = "Hello World"
    wrong_text_embedding = embedding_text(model, tokenizer, [wrong_text]).to(device).float()
    if args.normalize_embedding:
        wrong_text_embedding = normalize_embeddings(wrong_text_embedding)
    
    # wrong_text_embedding = wrong_text_embedding.repeat(len(eval_envs), 1)


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
        if args.subsample_video:
            traj_data = padding_video(traj_data, args.max_length)
        if args.normalize_embedding:
            traj_data = normalize_embeddings(traj_data)

        traj_data = traj_data.view(-1, 1024).unsqueeze(0).repeat(env_text_embedding.shape[0], 1, 1)
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        mask = torch.ones(traj_data.shape[1]).to(device).unsqueeze(0).repeat(traj_data.shape[0], 1)
        pred_class, two_step_class = self_attention_model(traj_data, triangle_mask, env_text_embedding, mask)

        batch_size, seq_len, _ = traj_data.size()
        if args.catagorical_progress:
            pred_class = torch.argmax(pred_class, dim = 1)
        else:
            pred_class = pred_class.squeeze(1)

        if args.two_step_training:
            pred_two_class = torch.argmax(two_step_class, dim = 1)
            pred_class = pred_two_class * pred_class

        pred_class = pred_class.view(batch_size, seq_len)

        wrong_pred_two_class, wrong_two_step_class = self_attention_model(traj_data, triangle_mask, wrong_text_embedding, mask)
        if args.catagorical_progress:
            wrong_pred_two_class = torch.argmax(wrong_pred_two_class, dim = 1)
        else:
            wrong_pred_two_class = wrong_pred_two_class.squeeze(1)

        if args.two_step_training:
            wrong_pred_two_class = torch.argmax(wrong_two_step_class, dim = 1)
            wrong_pred_two_class = wrong_pred_two_class * wrong_pred_two_class

        wrong_pred_two_class = wrong_pred_two_class.view(batch_size, seq_len)

        predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
        wrong_pred_two_class = np.array(wrong_pred_two_class.squeeze().detach().cpu().numpy())

        frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))

        figure = plt.figure()
        
        plt.plot(frame_index, predicted_classes, label="Correct Text", color="blue")
        plt.plot(frame_index, wrong_pred_two_class, label="Hello World", color="red")
        plt.xlabel("Frame Index")
        plt.ylabel("Class")
        plt.title(f"{env}")
        if args.catagorical_progress:
            plt.ylim(-1, 6)
        else:
            plt.ylim(-1, 1)
        # plt.savefig(f"progress_img/{env}.png")
        plt.legend()
        wandb.log({f"class_{set}/{env}": wandb.Image(figure)})
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
        embeddings = embeddings[::total_frames//num_frames]
    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        first_frame = embeddings[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_num, 1)
        embeddings = torch.cat([padding_frames, embeddings], dim=0)
    return embeddings







def plot_videos_class(model_name, self_attention_model, args):
    device = next(self_attention_model.parameters()).device
    model, processor, tokenizer = load_model(model_name)
    video_base_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos"
    video_idxs = ["1", "2"]
    diffs = ["GT", "all_fail", "close_succ", "success"]
    tasks = [
            "button_press",
            "button_press_wall", 
            "coffee_pull",
            "door_open",
            "drawer_close",
            "faucet_open",
            "handle_press_side",
            "handle_pull_side",
            "topdown", 
            "windowclose"
            ]

    texts = {
            "button_press": "Pressing button from side",
            "button_press_wall": "Pressing button from side",
            "coffee_pull": "Pulling cup",
            "door_open": "Opening door",
            "drawer_close": "Closing drawer",
            "faucet_open": "Opening faucet",
            "handle_press_side": "Pressing handle from side",
            "handle_pull_side": "Pulling handle from side",
            "topdown": "Pressing button from top",
            "windowclose": "Closing window"
            }


    for task in tasks:
        text = texts[task]
        text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
        if args.normalize_embedding:
            text_embeddings = normalize_embeddings(text_embeddings).float()


        for diff in diffs:
            for video_idx in video_idxs:
                gif_path = f"{video_base_path}/{task}/{diff}/{video_idx}.gif"
                # load gif
                frames = imageio.mimread(gif_path)
                frames = [frame[:,:,0:3] for frame in frames]
                if diff in ["all_fail", "close_succ"]:
                    frames = frames[:-1]
                # if args.subsample_video:
                #     frames = sample_video_frames(frames, num_frames = args.max_length)

                image_embeddings = []
                # select 32 frames
                for frame in frames:
                    image_embedding = embedding_image(model, processor, frame)
                    image_embeddings.append(image_embedding)
                image_embeddings = torch.stack(image_embeddings).to(device).float().squeeze(1)
                if args.normalize_embedding:
                    image_embeddings = normalize_embeddings(image_embeddings).float()
                if diff == "GT":
                    reverse_gt_embedding = image_embeddings.clone().flip(0)
                    cat_reverse = torch.cat([image_embeddings, reverse_gt_embedding[1:]], dim=0)
                if args.subsample_video:
                    image_embeddings = sample_embedding_frames(image_embeddings, num_frames = args.max_length)
                    reverse_gt_embedding = sample_embedding_frames(reverse_gt_embedding, num_frames = args.max_length)
                    cat_reverse = sample_embedding_frames(cat_reverse, num_frames = args.max_length)


                image_embeddings = image_embeddings.unsqueeze(0).float()
                triangle_mask = torch.tril(torch.ones(image_embeddings.shape[1], image_embeddings.shape[1])).to(device).unsqueeze(0).unsqueeze(0)
                mask = torch.ones(image_embeddings.shape[1]).to(device).unsqueeze(0)

                pred_class, two_step_class = self_attention_model(image_embeddings, triangle_mask, text_embeddings, mask)
                if args.catagorical_progress:
                    pred_class = torch.argmax(pred_class, dim = 1)
                else:
                    pred_class = pred_class.squeeze(1)


                predicted_output = pred_class.detach().cpu().numpy()
                # predicted_output = np.array(pred_class)
                frame_index = np.linspace(1, len(predicted_output), len(predicted_output))

                figure = plt.figure()
                plt.plot(frame_index, predicted_output )
                plt.xlabel("Frame Index")
                plt.ylabel("Similarity")
                plt.title(f"{task} {diff} {video_idx}")
                # set y axis range [-1,1]
                if args.catagorical_progress:
                    plt.ylim(-1, 6)
                else:
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


                if args.catagorical_progress:
                    gif_buffer = animate_video_with_rewards(frames, predicted_output, 15, _class = True)
                else:
                    gif_buffer = animate_video_with_rewards(frames, predicted_output, 15, _class = False)
                # mmrv = compute_mmrv(gt_index, predicted_output)
                # wandb.log({f"mmrv/{task}/{diff}_{video_idx}": mmrv})
                
                log_gif_to_wandb(gif_buffer, f"{task}/{diff}_{video_idx}")

                if diff == "GT":
                    reverse_gt_embedding = reverse_gt_embedding.unsqueeze(0).float()
                    pred_class, two_step_class = self_attention_model(reverse_gt_embedding, triangle_mask, text_embeddings, mask)
                    if args.catagorical_progress:
                        pred_class = torch.argmax(pred_class, dim = 1)
                    else:
                        pred_class = pred_class.squeeze(1)

                    predicted_output = pred_class.detach().cpu().numpy()
                    # predicted_output = np.array(pred_class)

                    frame_index = np.linspace(1, len(predicted_output), len(predicted_output))

                    figure = plt.figure()
                    plt.plot(frame_index, predicted_output )
                    plt.xlabel("Frame Index")
                    plt.ylabel("Similarity")
                    plt.title(f"Reverse {task} {diff} {video_idx}")
                    # set y axis range [-1,1]
                    if args.catagorical_progress:
                        plt.ylim(-1, 6)
                    else:
                        plt.ylim(-1, 1)

                    # plt.savefig(f"progress_img/{env}.png")
                    wandb.log({f"progress_video/{task}/reverse_{diff}_{video_idx}": wandb.Image(figure)})
                    plt.close()
                    print(f"progress_video/{task}/{diff}/reverse_{video_idx}")

                    
                    reverse_frames = frames[::-1]
                    predicted_output = np.stack(predicted_output)

                    gt_index = np.linspace(1, len(predicted_output), len(predicted_output))
                    act_index = np.argsort(predicted_output) + 1

                    # pearson correlation act_index
                    corr = np.corrcoef(act_index, gt_index)[0, 1]
                    wandb.log({f"corr/{task}/reverse_{diff}_{video_idx}": corr})


                    if args.catagorical_progress:
                        gif_buffer = animate_video_with_rewards(reverse_frames, predicted_output, 15, _class = True)
                    else:
                        gif_buffer = animate_video_with_rewards(reverse_frames, predicted_output, 15, _class = False)
                    
                    log_gif_to_wandb(gif_buffer, f"{task}/reverse_{diff}_{video_idx}")

                    cat_reverse = cat_reverse.unsqueeze(0).float()
                    triangle_mask = torch.tril(torch.ones(cat_reverse.shape[1], cat_reverse.shape[1])).to(device).unsqueeze(0).unsqueeze(0)
                    mask = torch.ones(cat_reverse.shape[1]).to(device).unsqueeze(0)

                    pred_class, two_step_class = self_attention_model(cat_reverse, triangle_mask, text_embeddings, mask)
                    if args.catagorical_progress:
                        pred_class = torch.argmax(pred_class, dim = 1)
                    else:
                        pred_class = pred_class.squeeze(1)

                    predicted_output = pred_class.detach().cpu().numpy()
                    # predicted_output = np.array(pred_class)

                    frame_index = np.linspace(1, len(predicted_output), len(predicted_output))

                    figure = plt.figure()
                    plt.plot(frame_index, predicted_output )
                    plt.xlabel("Frame Index")
                    plt.ylabel("Similarity")
                    plt.title(f"Concat {task} {diff} {video_idx}")
                    # set y axis range [-1,1]
                    if args.catagorical_progress:
                        plt.ylim(-1, 6)
                    else:
                        plt.ylim(-1, 1)

                    # plt.savefig(f"progress_img/{env}.png")
                    wandb.log({f"progress_video/{task}/concat_{diff}_{video_idx}": wandb.Image(figure)})
                    plt.close()
                    print(f"progress_video/{task}/{diff}/concat_{video_idx}")


                    cat_frames = np.concatenate([frames, reverse_frames[1:]], axis=0)
                    predicted_output = np.stack(predicted_output)
                    
                    gt_index = np.linspace(1, len(predicted_output), len(predicted_output))
                    act_index = np.argsort(predicted_output) + 1

                    # pearson correlation act_index
                    corr = np.corrcoef(act_index, gt_index)[0, 1]
                    wandb.log({f"corr/{task}/concat_{diff}_{video_idx}": corr})

                    

                    if args.catagorical_progress:
                        gif_buffer = animate_video_with_rewards(cat_frames, predicted_output, 15, _class = True)
                    else:
                        gif_buffer = animate_video_with_rewards(cat_frames, predicted_output, 15, _class = False)

                    log_gif_to_wandb(gif_buffer, f"{task}/concat_{diff}_{video_idx}")





        