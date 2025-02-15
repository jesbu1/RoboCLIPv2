import tensorflow_datasets as tfds
import random
from tqdm import tqdm
import os
from oxe_configs import OXE_DATASET_CONFIGS
import json
import numpy as np
import h5py
import torch
from transformers import AutoTokenizer, AutoModel
from clip_utils import (
    load_model,
    embedding_text,
    embedding_image,
    get_full_liv_embedding,
    dino_load_image,
    mean_pooling,
)
from PIL import Image

SAVE_H5_NAME = "droid_embeddings_dino.h5"  # name of the h5 file it'll be saved to
DEBUG = False # will use DROID_100
MAX_NUM_FRAMES_PER_EPISODE = 32
MAX_SAMPLES = float("inf")
TRAIN_SPLIT = "train"  # "test", droid training set doesn't exist

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
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
minilm_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
).to(device)


dataset_name = "droid"
if DEBUG:
    dataset_name += "_100"

# make a set to keep track of the tasks we've seen
tasks_seen = dict()
total_samples = 0
with h5py.File(SAVE_H5_NAME, "w") as f:
    dataset = tfds.load(dataset_name, data_dir="gs://gresearch/robotics", split=TRAIN_SPLIT)
    valid_samples_per_dataset = 0
    img_key_to_name = OXE_DATASET_CONFIGS[dataset_name.split("_")[0]][
        "image_obs_keys"
    ]  # dict mapping img_keys to the names of the images in OXE

    # get both ext camera left and ext camera right
    len_of_dataset = min(dataset.cardinality().numpy(), MAX_SAMPLES)
    views = [img_key_to_name["primary"], img_key_to_name["secondary"]]
    i = 0
    num_failures_in_a_row = 0
    # convert to iterator to be able to catch exception when failed to load an episode for any reason
    iter_dataset = iter(dataset)
    while True:
        try:
            episode = next(iter_dataset)
            # print progress
            print(
                f"------------------- Processing episode {i+1} out of {len_of_dataset} of dataset {dataset_name} -------------------"
            )
            i += 1
            # skip if we have already saved the video
            this_episode_name = f"{dataset_name}_ep{i}"

            episode_images_list = [[] for _ in range(len(views))]

            if valid_samples_per_dataset >= MAX_SAMPLES:
                break
            # task is the language instruction
            task = None
            for _, step in enumerate(episode["steps"]):
                # skip data loading if no lang
                if task is None or task == "":
                    for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
                        if key in step:
                            task = step[key].numpy().decode()
                            break
                # extract video
                for j, img_key in enumerate(views):
                    episode_images_list[j].append(step["observation"][img_key].numpy())
                    

            if task is None or task == '':
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

            for _, episode_images in enumerate(episode_images_list):
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
                        f[task].create_dataset("liv_lang_embedding", data=task_embedding)
                        individual_task_embedding = (
                            get_full_liv_embedding(liv_model, tokenizer, [task])
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # create a dataset with the embeddings
                        f[task].create_dataset(
                            "liv_lang_embedding_individual", data=individual_task_embedding
                        )

                        # create the minilm embeddings and save them
                        encoded_input = minilm_tokenizer(
                            [task], padding=False, truncation=True, return_tensors="pt"
                        ).to(device)

                        model_output = minilm_model(**encoded_input)
                        minlm_task_embedding = mean_pooling(
                            model_output, encoded_input["attention_mask"]
                        ).cpu().numpy()

                        f[task].create_dataset(
                            "minilm_lang_embedding", data=minlm_task_embedding
                        )

                        per_token_embeddings = model_output[0].cpu().numpy()

                        f[task].create_dataset(
                            "minilm_lang_embedding_individual", data=per_token_embeddings
                        )
                else:
                    tasks_seen[task] += 1

                # get the task group
                task_group = f[task]
                # get its length
                task_group_len = len(task_group.keys())
                # convert length to string
                task_group_len_str = str(task_group_len)

                # linspace to get the indices of the frames to sample
                indices = np.linspace(
                    0, len(episode_images) - 1, MAX_NUM_FRAMES_PER_EPISODE, dtype=int
                )
                # make sure there are no duplicates
                indices = list(set(indices))

                episode_images = [episode_images[i] for i in indices]

                # center crop 224x224
                if EMBEDDING_MODEL == "dinov2":
                    # batch it
                    with torch.inference_mode():
                        episode_images_dino = [
                            dino_load_image(img) for img in episode_images
                        ]
                        episode_images_dino = [torch.concatenate(episode_images_dino[i:i+DINO_BATCH_SIZE]) for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)]
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
                                Image.fromarray(ep_img.astype(np.uint8)),
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
