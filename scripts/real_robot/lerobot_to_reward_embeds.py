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
)

from PIL import Image
import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset


path = "/home/abrar/.cache/huggingface/lerobot/usc_koch_rewind/"
dataset_ids = glob.glob(path + "/*")

DATASET_IDS = [f"usc_koch_rewind/{os.path.basename(x)}" for x in dataset_ids]

SAVE_H5_NAME = (
    "usc_koch_rewind_reward_side_main.h5"  # name of the h5 file it'll be saved to
)
DEBUG = False  # will use DROID_100
MAX_NUM_FRAMES_PER_EPISODE = 32
# PRIMARY_IMAGE_KEY = "observation.images.main"
# PRIMARY_IMAGE_KEY = "observation.images.side"


# eval set. we will ignore these instructions
EVAL_TASKS = [
    "Put the blue cup on the red plate",
    "Separate the orange and blue cups",
    "Open the red trash bin",
    "Throw the banana away in the red trash bin",
    "Put the red tape in the box on the right",
]

model, processor, tokenizer = load_model("liv")
model = model.cuda()

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
                for key in image_keys:
                    episode_images_list[key].append(step[key].numpy())
                # episode_images_list[PRIMARY_IMAGE_KEY].append(
                #     step[PRIMARY_IMAGE_KEY].numpy()
                # )
                current_episode_idx = step["episode_index"]
                dataset_idx += 1
                if current_episode_idx != prev_episode_idx:
                    break

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
            # episode_images = episode_images_list[PRIMARY_IMAGE_KEY]
            for _, (img_key, episode_images) in enumerate(episode_images_list.items()):
                if task not in tasks_seen:
                    tasks_seen[task] = 1
                    if random.random() < 0.1:
                        print(f"Tasks seen so far: {tasks_seen.keys()}")
                    f.create_group(task)
                    task_embedding = (
                        embedding_text(model, tokenizer, [task]).detach().cpu().numpy()
                    )
                    # create a dataset with the embeddings
                    f[task].create_dataset("lang_embedding", data=task_embedding)
                    individual_task_embedding = (
                        get_full_liv_embedding(model, tokenizer, [task])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    # create a dataset with the embeddings
                    f[task].create_dataset(
                        "lang_embedding_individual", data=individual_task_embedding
                    )
                else:
                    tasks_seen[task] += 1

                # get the task group
                task_group = f[task]
                # get its length
                task_group_len = len(task_group.keys())
                # convert length to string
                task_group_len_str = str(task_group_len)

                embedding_list = []
                # linspace to get the indices of the frames to sample
                indices = np.linspace(
                    0, len(episode_images) - 1, MAX_NUM_FRAMES_PER_EPISODE, dtype=int
                )
                # make sure there are no duplicates
                indices = list(set(indices))

                episode_images = [episode_images[i] for i in indices]

                # center crop 224x224
                for ep_img in episode_images:
                    # NOTE: The transpose is for lerobot images only. Change if you are not using LeRobot
                    image_embeddings = (
                        embedding_image(
                            model,
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

                # create a dataset with the embeddings
                task_group.create_dataset(
                    task_group_len_str,
                    data=episode_image_embeddings,
                    # compression="gzip",
                    # compression_opts=9,
                )

                valid_samples_per_dataset += 1
                total_samples += 1
                print(f"Valid total samples: {total_samples} ")
        except StopIteration:
            break
        except Exception as e:
            print(f"Failed to load example with error: {e}")

print(f"Total samples: {total_samples}")
print(f"Tasks seen: {tasks_seen.keys()}")
