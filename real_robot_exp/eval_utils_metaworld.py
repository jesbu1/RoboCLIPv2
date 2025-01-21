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
task_subset = json.load(open("new_task_v2.json"))
train_tasks = task_subset["training_tasks"]
eval_tasks = task_subset["eval_tasks"]


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



def plot_progress_class(h5_file, set, self_attention_model, args):
    device = next(self_attention_model.parameters()).device
    if set == "train":
        eval_envs = train_tasks
    else:
        eval_envs = eval_tasks
    embedding_list = list()
    for env in eval_envs:
        env_name = env + "_text"
        embedding_list.append(np.asarray(h5_file[env_name]))
    embedding_list = np.stack(embedding_list)
    text_embeddings = torch.tensor(embedding_list).to(device).float()

    if args.normalize_embedding:
        text_embeddings = normalize_embeddings(text_embeddings)

    # wrong_text = "Hello World"
    wrong_text_embedding = np.asarray(h5_file["clean-table_text"])
    wrong_text_embedding = torch.tensor(wrong_text_embedding).to(device).float().unsqueeze(0)
    if args.normalize_embedding:
        wrong_text_embedding = normalize_embeddings(wrong_text_embedding)

    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        
        traj_data = np.asarray(h5_file[env])
        env_text_embedding = text_embeddings[i:i+1]
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
            two_step_class = two_step_class.squeeze(0)
            pred_two_class = torch.argmax(two_step_class, dim = 1)
            pred_class = pred_two_class * pred_class.squeeze(0).squeeze(1)
        # pred_class = pred_class.view(batch_size, seq_len)

        wrong_pred_two_class, wrong_two_step_class = self_attention_model(traj_data, triangle_mask, wrong_text_embedding, mask)
        if args.catagorical_progress:
            wrong_pred_two_class = torch.argmax(wrong_pred_two_class, dim = 1)
        else:
            wrong_pred_two_class = wrong_pred_two_class.squeeze(1)

        if args.two_step_training:
            wrong_two_step_class = wrong_two_step_class.squeeze(0)
            wrong_pred_two_class = torch.argmax(wrong_two_step_class, dim = 1)
            wrong_pred_two_class = wrong_pred_two_class * wrong_pred_two_class

        # wrong_pred_two_class = wrong_pred_two_class.view(batch_size, seq_len)

        predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
        wrong_pred_two_class = np.array(wrong_pred_two_class.squeeze().detach().cpu().numpy())

        frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))

        figure = plt.figure()
        
        plt.plot(frame_index, predicted_classes, label="Correct Text", color="blue")
        plt.plot(frame_index, wrong_pred_two_class, label="Cleaning the table", color="red")
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





        