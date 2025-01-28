import tensorflow_datasets as tfds
import random
from tqdm import tqdm
import os
from oxe_configs import OXE_DATASET_CONFIGS
import json
import numpy as np
import h5py
from clip_utils import (
    load_model,
    embedding_text,
    embedding_image,
    get_full_liv_embedding,
)
from PIL import Image

SAVE_H5_NAME = "droid_embeddings.h5"  # name of the h5 file it'll be saved to
DEBUG = True  # willuse DROID_100
MAX_NUM_FRAMES_PER_EPISODE = 128
TRAIN_SPLIT = "train"  # "test"

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "natural_language_instruction",
    "instruction",
    "language_instruction",
]


model, processor, tokenizer = load_model("liv")
model = model.cuda()

dataset_name = "droid"
if DEBUG:
    dataset_name += "_100"

# make a set to keep track of the tasks we've seen
tasks_seen = dict()
total_samples = 0
with h5py.File(SAVE_H5_NAME, "w") as f:
    dataset = tfds.load(dataset_name, data_dir="gs://gresearch/robotics", split="train")
    n_samples = 10 if DEBUG else 1000000000000000000
    valid_samples_per_dataset = 0
    img_key_to_name = OXE_DATASET_CONFIGS[dataset_name][
        "image_obs_keys"
    ]  # dict mapping img_keys to the names of the images in OXE

    # get the image key that matches "primary" to get the main camera view
    for img_key in [img_key_to_name["primary"], img_key_to_name["secondary"]]:
        i = 0
        len_of_dataset = min(dataset.cardinality().numpy(), n_samples)
        # convert to iterator to be able to catch exception when failed to load an episode for any reason
        dataset = iter(dataset)
        num_failures_in_a_row = 0
        while True:
            try:
                episode = next(dataset)
                # print progress
                print(
                    f"------------------- Processing episode {i+1} out of {len_of_dataset} of dataset {dataset_name} -------------------"
                )
                # skip if we have already saved the video
                this_episode_name = f"{dataset_name}_ep{i}"

                episode_images = []

                if valid_samples_per_dataset >= n_samples:
                    break
                # task is the language instruction
                task = None
                for _, step in enumerate(episode["steps"]):
                    # skip data loading if no lang
                    for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
                        if key in step["observation"]:
                            if dataset_name == "language_table":
                                task = step["observation"][key].numpy()
                                task = bytes(task[np.where(task != 0)].tolist()).decode(
                                    "utf-8"
                                )
                            else:
                                task = step["observation"][key].numpy().decode()
                            break
                        elif key in step:
                            task = step[key].numpy().decode()
                            break
                    # extract video
                    episode_images.append(step["observation"][img_key].numpy())

                if task is None:
                    print(
                        f"Skipping episode {i + 1} of dataset {dataset_name} as the task is None or empty."
                    )
                    print(
                        f"Keys in step: {step.keys()} and keys in observation: {step['observation'].keys()}"
                    )
                    continue

                # process task name to capitalize the first letter
                task = task.capitalize()
                # process task name to not have a period at the end and strip other punctuation
                task = task.strip(" .,!?-_")

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
                    image_embeddings = (
                        embedding_image(
                            model, processor, Image.fromarray(ep_img.astype(np.uint8))
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
                i += 1
                total_samples += 1
                print(f"Valid total samples: {total_samples} ")
            except StopIteration:
                break
            except Exception as e:
                print(f"Failed to load dataset {dataset_name} with error: {e}")
                num_failures_in_a_row += 1
                if num_failures_in_a_row > 10:
                    break
