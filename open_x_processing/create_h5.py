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

TFDS_PATH = "/data/shared/openx_rlds_data"
SAVE_H5_NAME = "openx_embeddings_lang_table.h5"  # name of the h5 file it'll be saved to
DEBUG = False  # will only make 10 per dataset
SPECIFIC_TASKS = "language_table,austin_sirius_dataset_converted_externally_to_rlds,austin_buds_dataset_converted_externally_to_rlds,ucsd_kitchen_dataset_converted_externally_to_rlds,stanford_hydra_dataset_converted_externally_to_rlds,iamlab_cmu_pickup_insert_converted_externally_to_rlds,cmu_stretch,berkeley_fanuc_manipulation,berkeley_autolab_ur5,bridge,bc_z,fractal20220817_data,jaco_play"
MAX_NUM_FRAMES_PER_EPISODE = 128
TRAIN_SPLIT = "train"  # "test"
MAX_EPISODES_FOR_LANG_TABLE = 25000

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "natural_language_instruction",
    "instruction",
    "language_instruction",
]


model, processor, tokenizer = load_model("liv")
model = model.cuda()


# dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]
# load dataset names from the TFDS path
dataset_names = os.listdir(TFDS_PATH)
print(dataset_names)

if SPECIFIC_TASKS is not None:
    # overwrite the dataset_names with the specific tasks
    dataset_names = SPECIFIC_TASKS.split(",")

# make a set to keep track of the tasks we've seen
tasks_seen = dict()
total_samples = 0
with h5py.File(SAVE_H5_NAME, "w") as f:
    for dataset_name in tqdm(dataset_names):
        try:
            dataset = tfds.load(dataset_name, data_dir=TFDS_PATH, split=TRAIN_SPLIT)
        except ValueError as e:
            print(f"Failed to load dataset {dataset_name}: \n{e}")
            continue
        n_samples = 10 if DEBUG else 1000000000000000000
        valid_samples_per_dataset = 0
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name][
            "image_obs_keys"
        ]  # dict mapping img_keys to the names of the images in OXE

        # get the image key that matches "primary" to get the main camera view
        primary_img_key = img_key_to_name["primary"]
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
                    episode_images.append(step["observation"][primary_img_key].numpy())

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
                if (
                    valid_samples_per_dataset > MAX_EPISODES_FOR_LANG_TABLE
                    and dataset_name == "language_table"
                ):
                    # control cause language table has 444k trajs
                    break
            except StopIteration:
                break
            except Exception as e:
                print(f"Failed to load dataset {dataset_name} with error: {e}")
                num_failures_in_a_row += 1
                if num_failures_in_a_row > 10:
                    break
