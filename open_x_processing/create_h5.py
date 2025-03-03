import tensorflow_datasets as tfds
import torchvision.transforms as T
import torch
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
    dino_load_image,
    mean_pooling,
)
from PIL import Image
from transformers import AutoTokenizer, AutoModel

TFDS_PATH = "/data/shared/openx_rlds_data"
TRAIN_SPLIT = "train"  # "train"
SAVE_H5_DIR = f"{TRAIN_SPLIT}_dataset_embeddings"  # directory to save individual h5 files
DEBUG = False  # will only make 10 per dataset
SPECIFIC_TASKS = "language_table,austin_sirius_dataset_converted_externally_to_rlds,austin_buds_dataset_converted_externally_to_rlds,ucsd_kitchen_dataset_converted_externally_to_rlds,stanford_hydra_dataset_converted_externally_to_rlds,iamlab_cmu_pickup_insert_converted_externally_to_rlds,cmu_stretch,berkeley_fanuc_manipulation,berkeley_autolab_ur5,bridge_v2,bc_z,fractal20220817_data,jaco_play"
MAX_NUM_FRAMES_PER_EPISODE = 32
MAX_EPISODES_FOR_LANG_TABLE = 10000

# Create directory for saving h5 files if it doesn't exist
os.makedirs(SAVE_H5_DIR, exist_ok=True)

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "natural_language_instruction",
    "instruction",
    "language_instruction",
]
EMBEDDING_MODEL = "dinov2"
DINO_BATCH_SIZE = 32
assert EMBEDDING_MODEL in ["liv", "dinov2"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dinov2_vits14 = dinov2_vits14.to(device)
liv_model, processor, tokenizer = load_model("liv")
liv_model = liv_model.to(device)

# also load mini_lm sentence embeddings from sentence_transformers
minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)


# dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]
# load dataset names from the TFDS path
dataset_names = os.listdir(TFDS_PATH)
print(dataset_names)

if SPECIFIC_TASKS is not None:
    # overwrite the dataset_names with the specific tasks
    dataset_names = SPECIFIC_TASKS.split(",")

# make a set to keep track of the tasks we've seen
total_samples = 0
for dataset_name in tqdm(dataset_names):
    # Create a new h5 file for each dataset
    h5_file_path = os.path.join(
        SAVE_H5_DIR, f"{dataset_name}_{TRAIN_SPLIT}_embeddings.h5"
    )
    tasks_seen = dict()  # Reset tasks_seen for each dataset

    with h5py.File(h5_file_path, "w") as f:
        try:
            if TRAIN_SPLIT == "test":
                try:
                    dataset = tfds.load(
                        dataset_name, data_dir=TFDS_PATH, split=TRAIN_SPLIT
                    )
                except ValueError as e:
                    dataset = tfds.load(dataset_name, data_dir=TFDS_PATH, split="val")
            else:
                dataset = tfds.load(dataset_name, data_dir=TFDS_PATH, split=TRAIN_SPLIT)
        except ValueError as e:
            print(f"Failed to load dataset {dataset_name}: \n{e}")
            continue
        n_samples = 10 if DEBUG else 1000000000000000000
        valid_samples_per_dataset = 0
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name][
            "image_obs_keys"
        ]  # dict mapping img_keys to the names of the images in OXE

        # get all non-None image keys except "wrist"
        valid_img_values = [
            v for k, v in img_key_to_name.items() if v is not None and k != "wrist"
        ]
        print(f"Valid image values for dataset {dataset_name}: {valid_img_values}")

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

                episode_images = dict()
                for key in valid_img_values:
                    episode_images[key] = []

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
                    # extract images from all valid keys
                    for img_key in valid_img_values:
                        if img_key in step["observation"]:
                            episode_images[img_key].append(
                                step["observation"][img_key].numpy()
                            )

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

                # get the task group
                task_group = f[task]
                # get its length
                task_group_len = len(task_group.keys())
                # convert length to string
                task_group_len_str = str(task_group_len)

                # Process each image key separately
                for img_key in valid_img_values:
                    if len(episode_images[img_key]) == 0:
                        print(f"Skipping {img_key} as no images were found")
                        continue
                    if np.all(episode_images[img_key] == 0):
                        print(f"Skipping {img_key} as all images are 0.")
                        continue

                    # linspace to get the indices of the frames to sample
                    indices = np.linspace(
                        0,
                        len(episode_images[img_key]) - 1,
                        MAX_NUM_FRAMES_PER_EPISODE,
                        dtype=int,
                    )
                    # make sure there are no duplicates
                    indices = sorted(list(set(indices)))

                    sampled_images = [episode_images[img_key][i] for i in indices]

                    # center crop 224x224
                    if EMBEDDING_MODEL == "dinov2":
                        with torch.inference_mode():
                            # batch it
                            episode_images_dino = [
                                dino_load_image(img) for img in sampled_images
                            ]
                            episode_images_dino = [
                                torch.concatenate(
                                    episode_images_dino[i : i + DINO_BATCH_SIZE]
                                )
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
                            episode_image_embeddings = np.concatenate(embedding_list)
                    else:
                        embedding_list = []
                        for ep_img in sampled_images:
                            image_embeddings = (
                                embedding_image(
                                    liv_model,
                                    processor,
                                    Image.fromarray(ep_img.astype(np.uint8)),
                                )
                                .squeeze()
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            embedding_list.append(image_embeddings)
                        episode_image_embeddings = np.array(embedding_list)
                    # create a dataset with the embeddings using the image key as part of the name
                    task_group.create_dataset(
                        f"{task_group_len_str}_{img_key}",
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
