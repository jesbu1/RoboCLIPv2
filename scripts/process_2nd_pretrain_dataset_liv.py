import argparse
import torch
import torch as th
import h5py
import torch.nn.functional as F

import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModel
from new_task_annotation_v2 import train_gt_annotation, generated_gt_annotation
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from models_pe import ClassProgressTransformer as RewindModel

from models.reward_model.liv_reward_model import LIVRewardModel




# def load_rewind_model():
#     # load the model
#     model_dict = torch.load(model_path)
#     args = model_dict['args']
#     model = RewindModel(
#             args=args,
#             video_dim=768,  # Original video embedding dimension
#             text_dim=384,   # Original text embedding dimension
#             hidden_dim=512  # Common dimension for transformer processing
#         ).to(device)
#     model.load_state_dict(model_dict['ema_model'])
#     return model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )



def load_model():
    minilm_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    )
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
        device
    )
    return minilm_model, minilm_tokenizer



def embedding_lang(ann, minilm_model, minilm_tokenizer):
    lang_embeddings = list()
    for task in ann:
        encoded_input = minilm_tokenizer(
            [task], padding=False, truncation=True, return_tensors="pt"
        ).to(device)

        model_output = minilm_model(**encoded_input)
        minlm_task_embedding = (
            mean_pooling(model_output, encoded_input["attention_mask"])
            .cpu()
            .detach()
            .numpy()
        )
        lang_embeddings.append(minlm_task_embedding)
    lang_embeddings = np.concatenate(lang_embeddings, axis=0)
    return lang_embeddings


dino_transform_image = T.Compose(
    [T.ToTensor(), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
)

def dino_load_image(image):
    img = Image.fromarray(image)

    transformed_img = dino_transform_image(img)[:3].unsqueeze(0)

    return transformed_img


dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dinov2_vits14 = dinov2_vits14.to(device)
DINO_BATCH_SIZE = 200

def embedding_image(sampled_images):
    with torch.inference_mode():
        # batch it
        episode_images_dino = [
            dino_load_image(img) for img in sampled_images
        ]
        episode_images_dino = [torch.concatenate(episode_images_dino[i : i + DINO_BATCH_SIZE])
        for i in range(
            0, len(episode_images_dino), DINO_BATCH_SIZE
        )
        ]
        embedding_list = []
        for batch in episode_images_dino:
            episode_image_embeddings = (
                dinov2_vits14(batch.to(device))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            embedding_list.append(episode_image_embeddings)
        try:
            episode_image_embeddings = np.concatenate(embedding_list)
        except:
            for i in range(len(embedding_list)):
                print("embedding_list[i].shape", embedding_list[i].shape)
            import pdb ; pdb.set_trace()
            a = 0
    return episode_image_embeddings

# def subsample_video(video_frames, max_length = 16):
#     video_length = len(video_frames)
#     if type(video_frames) == np.ndarray:
#         video_frames = th.tensor(video_frames)
#     if video_length < max_length:
#         # padding last frame
#         padding_length = max_length - video_length
#         # first_frame = video_frames[0].unsqueeze(0)
#         last_frame = video_frames[-1].unsqueeze(0)
#         padding_frames = last_frame.repeat(padding_length, 1)
#         video_frames = th.cat([video_frames, padding_frames], dim=0)
#         # video_frames = th.cat([padding_frames, video_frames], dim=0)
    
#     elif video_length > max_length:
#         frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
#         video_frames = video_frames[frame_idx]

#     return video_frames



def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "metaworld_5_Demo_PosEmb_View_side_OpenX_Rewind_ratio_0.8_EMA_momentum_0.3_End_Rewind_ratio_0.1/model_20.pth"
    reward_model = LIVRewardModel(
            model_load_path='snapshot_10000.pt',
            use_pca=False,
            attention_heads=4,
            pca_model_dir=None,
            batch_size=64,
            success_bonus=200,
            last_frame_reward_only = True,
        )

    dataset_path = "../open_x_processing/metaworld_pretraining_dataset_10.h5"
    h5_file = h5py.File(dataset_path, "a")
    minilm_model, minilm_tokenizer = load_model()
    # add annotation
   
    merge_gt_annotation = {}
    merge_gt_annotation_embedding = {}
    merge_gt_annotation_string = {}

#     if args.reward_type == "rewind":
#         model = load_rewind_model()
#         model.eval()
#         model.to(device)
    if "text_embedding" not in h5_file[list(h5_file.keys())[-1]]:
        print("text_embedding not exists, start to generate minilm embedding")
        for key in train_gt_annotation.keys():
            merge_gt_annotation[key] = [train_gt_annotation[key]] + generated_gt_annotation[key]
            lang_embeddings = embedding_lang(merge_gt_annotation[key], minilm_model, minilm_tokenizer)
            merge_gt_annotation_embedding[key] = lang_embeddings
            merge_gt_annotation_string[key] = np.asarray(merge_gt_annotation[key], dtype=h5py.string_dtype(encoding='utf-8'))
            if "text_embedding" in h5_file[key]:
                del h5_file[key]["text_embedding"]
            if "text_string" in h5_file[key]:
                del h5_file[key]["text_string"]
            h5_file[key].create_dataset(
                "text_embedding", data=merge_gt_annotation_embedding[key]
            )
            h5_file[key].create_dataset(
                "text_string", data=merge_gt_annotation_string[key]
            )
    else:
        print("text_embedding already exists")
    

#     # generate pretraining dataset structure same with abrar
    pre_training_dataset = f"metaworld_policy_pretrain_dataset_vlc_5.h5" # this is the dataset for pretraining
#     if args.reward_type == "rewind":
#         pre_training_dataset = f"metaworld_policy_pretrain_dataset_rewind_dense_10.h5"
    pre_training_h5_file = h5py.File(pre_training_dataset, "w")

    group_keys = ["rewards", 
                  "lang_embedding", 
                  "policy_lang_embedding", 
                #   "img", 
                  "img_embedding", 
                  "timesteps",  
                  "text_string", 
                  "env_id", 
                  "state", 
                  "next_state", 
                    "done",
                  "action"]

    # traj_keys = ["0", "1", "10", "11", "12", "13", "14", "2", "3", "4"]
    traj_keys = ["0", "1", "10", "11", "12"]
    

    total_timesteps = 0
    for key in h5_file.keys():
        group = h5_file[key]
        for subkey in traj_keys:
            total_timesteps += len(group[traj_keys[0]]["action"])
    total_timesteps = total_timesteps * 5 # each task has 6 annotations


    pre_training_h5_file.create_dataset("lang_embedding", (total_timesteps, 384), dtype="float32")
    pre_training_h5_file.create_dataset("policy_lang_embedding", (total_timesteps, 384), dtype="float32")
    pre_training_h5_file.create_dataset("img_embedding", (total_timesteps, 768), dtype="float32")
    pre_training_h5_file.create_dataset("timesteps", (total_timesteps,), dtype="int32")
    pre_training_h5_file.create_dataset("text_string", (total_timesteps,), dtype=h5py.string_dtype(encoding='utf-8'))
    pre_training_h5_file.create_dataset("env_id", (total_timesteps,), dtype=h5py.string_dtype(encoding='utf-8'))
    pre_training_h5_file.create_dataset("state", (total_timesteps, 39), dtype="float32")
    pre_training_h5_file.create_dataset("next_state", (total_timesteps, 39), dtype="float32")
    pre_training_h5_file.create_dataset("action", (total_timesteps, 4), dtype="float32")
    pre_training_h5_file.create_dataset("done", (total_timesteps,), dtype="float32")
    pre_training_h5_file.create_dataset("rewards", (total_timesteps,), dtype="float32")


    current_step = 0
    for key in tqdm(h5_file.keys()):
        group = h5_file[key]
        for traj_id in traj_keys:
            import pdb ; pdb.set_trace()
            traj_imgs = np.asarray(group[traj_id]['img'])
            traj_img_embeddings = embedding_image(traj_imgs)
            traj_imgs_for_liv = np.expand_dims(traj_imgs, axis=0)
            traj_imgs_for_liv = reward_model.encode_images(traj_imgs_for_liv)
            

            states = np.asarray(group[traj_id]['state'])
            next_states = np.asarray(group[traj_id]['next_state'])
            actions = np.asarray(group[traj_id]['action'])
            
            done = np.asarray(group[traj_id]['done'])
            env_id = np.asarray(key, dtype=h5py.string_dtype(encoding='utf-8'))

            text_embedding_all = np.asarray(h5_file[key]['text_embedding'])
            text_liv_embedding_all = np.asarray(h5_file[key]['text_embedding_liv'])
            text_string_all = np.asarray(h5_file[key]['text_string'])
            rewards = np.asarray(group[traj_id]['reward'])



            for i in range(len(text_embedding_all)):
                text_embedding = text_embedding_all[i]
                text_string = text_string_all[i]
                text_liv_embedding = text_liv_embedding_all[i]

                for j in range(len(states)):
                    pre_training_h5_file["lang_embedding"][current_step] = text_embedding
                    pre_training_h5_file["policy_lang_embedding"][current_step] = text_embedding
                    pre_training_h5_file["img_embedding"][current_step] = traj_img_embeddings[j]
                    
                    pre_training_h5_file["timesteps"][current_step] = j
                    pre_training_h5_file["text_string"][current_step] = text_string
                    pre_training_h5_file["env_id"][current_step] = env_id
                    pre_training_h5_file["state"][current_step] = states[j]
                    pre_training_h5_file["next_state"][current_step] = next_states[j]
                    pre_training_h5_file["action"][current_step] = actions[j]
                    pre_training_h5_file["done"][current_step] = done[j]

                    liv_reward = reward_model.calculate_rewards(
                        np.expand_dims(text_liv_embedding, axis=0),
                        np.expand_dims(traj_imgs_for_liv[:j+1], axis=0),
                    )

                    pre_training_h5_file["rewards"][current_step] = liv_reward
                    current_step += 1



    print("current_step", current_step, "total_timesteps", total_timesteps, "sum reward", np.sum(pre_training_h5_file["rewards"]))
    print("max reward", np.max(pre_training_h5_file["rewards"]), "min reward", np.min(pre_training_h5_file["rewards"]))
    pre_training_h5_file.close()

    
#     # now we have 5 annotations, so for each (state, action) pair, we will save 5 times, and each time we will save with different annotation


    


#     # rewards = output_file["rewards"]
#     # lang_embeds = output_file["lang_embedding"]
#     # policy_lang_embeds = output_file["policy_lang_embedding"]
#     # img_embeds = output_file["img_embedding"]
#     # timesteps = output_file["timesteps"]
#     # img_dataset = output_file["img"]



if __name__ == "__main__":

    main()
            




