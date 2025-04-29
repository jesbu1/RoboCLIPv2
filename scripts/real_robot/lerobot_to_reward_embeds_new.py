import random
from tqdm import tqdm
import os
import json
import numpy as np
import h5py
from reward_model.clip_utils import (
    load_model,
    embedding_text,
    embedding_image,
    get_full_liv_embedding,
    dino_load_image,
    mean_pooling,
)

import torch
from PIL import Image
import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from transformers import AutoTokenizer, AutoModel
import cv2
from matplotlib import pyplot as plt

from usc_episode_rescaling import rescaling_dict

path = "/home/abrar/.cache/huggingface/lerobot/usc_koch_rewind/"
dataset_ids = glob.glob(path + "/*")

DATASET_IDS = [f"usc_koch_rewind/{os.path.basename(x)}" for x in dataset_ids]

# filter it so only tasks with "_2" is in them
# NOTE: This is only to train on the newer data!
DATASET_IDS = [x for x in DATASET_IDS if "_2" in x]

SAVE_H5_NAME = (
    "usc_koch_rewind_dino_reward_side_new.h5"  # name of the h5 file it'll be saved to
)
DEBUG = False  # will use DROID_100
FRAMES_TO_START_FROM = 128
MAX_NUM_FRAMES_PER_EPISODE = 32
# PRIMARY_IMAGE_KEY = "observation.images.main"
PRIMARY_IMAGE_KEY = "observation.images.side"


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
EMBEDDING_MODEL = "dinov2"
DINO_BATCH_SIZE = 32
assert EMBEDDING_MODEL in ["liv", "dinov2"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dinov2_vits14 = dinov2_vits14.to(device)
liv_model, processor, tokenizer = load_model("liv")
liv_model = liv_model.cuda(device)

# also load mini_lm sentence embeddings from sentence_transformers
minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)


# make a set to keep track of the tasks we've seen
tasks_seen = dict()
total_samples = 0
with h5py.File(SAVE_H5_NAME, "w") as f:
    n_samples = 1000000000000000000
    valid_samples_per_dataset = 0

    # Open dataset
    image_transforms = None
    delta_timestamps = None
    video_backend = "pyav"
    if isinstance(DATASET_IDS, str):
        dataset = LeRobotDataset(
            DATASET_IDS,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=video_backend,
            local_files_only=True,
        )
    else:
        dataset = MultiLeRobotDataset(
            DATASET_IDS,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=video_backend,
            local_files_only=True,
        )

    image_keys = ["observation.images.main", "observation.images.side"]

    # get both ext camera left and ext camera right
    len_of_dataset = min(len(dataset), n_samples)
    i = 0
    dataset_idx = 0
    num_failures_in_a_row = 0
    # convert to iterator to be able to catch exception when failed to load an episode for any reason
    while dataset_idx < len_of_dataset:
        try:
            # print progress
            print(
                f"------------------- Processing episode {i + 1} out of {len_of_dataset} -------------------"
            )
            i += 1

            # Let's collect all information for 1 episode
            episode_images_list = {}
            for key in image_keys:
                episode_images_list[key] = []

            task = None

            prev_episode_idx = dataset[dataset_idx]["episode_index"]
            while True:
                step = dataset[dataset_idx]
                task = dataset._datasets[step["dataset_index"]].meta.episodes[
                    step["episode_index"]
                ]["tasks"][0]
                repo_id = dataset._datasets[step["dataset_index"]].meta.repo_id
                for key in image_keys:
                    episode_images_list[key].append(step[key].numpy())

                dataset_idx += 1
                if dataset_idx >= len_of_dataset:
                    break
                if dataset[dataset_idx]["episode_index"] != prev_episode_idx:
                    break
            episode_images = np.array(episode_images_list[PRIMARY_IMAGE_KEY])

            if valid_samples_per_dataset >= n_samples:
                break

            if task is None or task == "":
                print(
                    f"Skipping episode {i + 1} of dataset as the task is None or empty."
                )
                print(
                    f"Keys in step: {step.keys()} and keys in observation: {step['observation'].keys()}"
                )
                continue

            # process task name to capitalize the first letter
            task = task.capitalize()
            # process task name to not have a period at the end and strip other punctuation
            task = task.strip(" .,!?-_")
            # for _, (img_key, episode_images) in enumerate(episode_images_list.items()):
            if True:
                if task not in tasks_seen:
                    with torch.inference_mode():
                        tasks_seen[task] = 1
                        if random.random() < 0.01:
                            print(f"Tasks seen so far: {tasks_seen.keys()}")
                        f.create_group(task)
                        task_embedding = (
                            embedding_text(liv_model, tokenizer, [task])
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # create a dataset with the embeddings
                        f[task].create_dataset(
                            "liv_lang_embedding", data=task_embedding
                        )
                        individual_task_embedding = (
                            get_full_liv_embedding(liv_model, tokenizer, [task])
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # create a dataset with the embeddings
                        f[task].create_dataset(
                            "liv_lang_embedding_individual",
                            data=individual_task_embedding,
                        )

                        # create the minilm embeddings and save them
                        encoded_input = minilm_tokenizer(
                            [task], padding=False, truncation=True, return_tensors="pt"
                        ).to(device)

                        model_output = minilm_model(**encoded_input)
                        minlm_task_embedding = (
                            mean_pooling(model_output, encoded_input["attention_mask"])
                            .cpu()
                            .numpy()
                        )

                        f[task].create_dataset(
                            "minilm_lang_embedding", data=minlm_task_embedding
                        )

                        per_token_embeddings = model_output[0].cpu().numpy()

                        f[task].create_dataset(
                            "minilm_lang_embedding_individual",
                            data=per_token_embeddings,
                        )
                else:
                    tasks_seen[task] += 1

                print("current task is " + task)

                # get the task group
                task_group = f[task]
                # get its length
                task_group_len = len(task_group.keys())
                # convert length to string
                task_group_len_str = str(task_group_len)

                # rescale temporally with rescaling_dict
                if repo_id in rescaling_dict:
                    episode_images = episode_images[
                        : int(len(episode_images) * rescaling_dict[repo_id])
                    ]

                embedding_list = []
                # linspace to get the indices of the frames to sample
                indices = np.linspace(
                    0, len(episode_images) - 1, MAX_NUM_FRAMES_PER_EPISODE, dtype=int
                )
                # make sure there are no duplicates
                indices = sorted(list(set(indices)))

                episode_images = [episode_images[i] for i in indices]

                # # After collecting all frames for an episode, display the PRIMARY_IMAGE_KEY frames in a grid
                primary_frames = episode_images
                num_frames = len(primary_frames)

                # Convert frames to correct format
                primary_frames = [
                    (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
                    for frame in primary_frames
                ]

                # Create a grid layout (e.g., 4x8 for 32 frames)
                # rows, cols = 4, 8
                # fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
                # fig.suptitle(f"All frames for episode {prev_episode_idx}")

                # # Plot each frame
                # for idx in range(rows * cols):
                #     ax = axes[idx // cols, idx % cols]
                #     if idx < num_frames:
                #         ax.imshow(primary_frames[idx])
                #     ax.axis("off")
                #     # if idx < num_frames:
                #     #     ax.set_title(f'Frame {idx}')

                # plt.tight_layout()
                # plt.pause(2)
                # plt.close()

                if EMBEDDING_MODEL == "dinov2":
                    # batch it

                    with torch.inference_mode():
                        episode_images_dino = [
                            dino_load_image(
                                (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                            )
                            for img in episode_images
                        ]
                        episode_images_dino = [
                            torch.concatenate(
                                episode_images_dino[i : i + DINO_BATCH_SIZE]
                            )
                            for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)
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
                        episode_image_embeddings = np.concatenate(embedding_list)
                else:
                    embedding_list = []
                    for ep_img in episode_images:
                        image_embeddings = (
                            embedding_image(
                                liv_model,
                                processor,
                                Image.fromarray(
                                    (ep_img * 255).astype(np.uint8).transpose(1, 2, 0)
                                ),
                            )
                            .squeeze()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    embedding_list.append(image_embeddings)
                    episode_image_embeddings = np.array(embedding_list)

                # Check for embedding differences over time. We want to find the cutoff point where the embeddings stop changing
                # offset_embeddings = episode_image_embeddings[1:] - episode_image_embeddings[:-1]
                # diff = np.abs(offset_embeddings).mean(axis=1)
                # # show the top 10% of the differences
                # cutoff = np.percentile(diff, 1)
                # # now find the first frame where the difference is below the cutoff
                # first_frame = np.where(diff <= cutoff)[0][0]

                # # print task name
                # print("Task: ", task)

                # print("First frame where embeddings stop changing: ", first_frame, "out of", len(episode_image_embeddings))

                # episode_image_embeddings = episode_image_embeddings[:first_frame]

                # # now downsample

                # indices = np.linspace(
                #     0, len(episode_image_embeddings) - 1, MAX_NUM_FRAMES_PER_EPISODE, dtype=int
                # )
                # episode_image_embeddings = episode_image_embeddings[indices]

                # # We want to now downsample these embeddings to the max frames

                # create a dataset with the embeddings
                task_group.create_dataset(
                    task_group_len_str,
                    data=episode_image_embeddings,
                    # compression="gzip",
                    # compression_opts=9,
                )

                # save the images too
                # task_group.create_dataset(
                #     task_group_len_str + "_images",
                #     data=np.array(episode_images),
                #     # compression="gzip",
                #     # compression_opts=9,
                # )

                valid_samples_per_dataset += 1
                total_samples += 1
                print(f"Valid total samples: {total_samples} ")
        except StopIteration:
            break
        except Exception as e:
            print(f"Failed to load example with error: {e}")

print(f"Total samples: {total_samples}")
print(f"Tasks seen: {tasks_seen.keys()}")
