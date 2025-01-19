import tensorflow_datasets as tfds
from tqdm import tqdm
import os
from oxe_configs import OXE_DATASET_CONFIGS
import json
import numpy as np
import h5py
from clip_utils import load_model, embedding_text, embedding_image
from PIL import Image

TFDS_PATH = "/data/shared/openx_rlds_data"
SAVE_H5_NAME = "openx_embeddings.h5"  # name of the h5 file it'll be saved to
DEBUG = False # will only make 5 per dataset
SPECIFIC_TASKS = None #"bridge"

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "natural_language_instruction",
    "instruction",
    "language_instruction",
]


model, processor, tokenizer = load_model("liv")
model = model.cuda()


#dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]
# load dataset names from the TFDS path
dataset_names = os.listdir(TFDS_PATH)
print(dataset_names)

if SPECIFIC_TASKS is not None:
    # overwrite the dataset_names with the specific tasks
    dataset_names = SPECIFIC_TASKS.split(",")

# make a set to keep track of the tasks we've seen
tasks_seen = set()
total_samples = 0
with h5py.File(SAVE_H5_NAME, "w") as f:
    for dataset_name in tqdm(dataset_names):
        dataset = tfds.load(dataset_name, data_dir=TFDS_PATH, split="train")
        n_samples = 5 if DEBUG else 1000000000000000000
        valid_samples_per_dataset = 0
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name][
            "image_obs_keys"
        ]  # dict mapping img_keys to the names of the images in OXE

        # get the image key that matches "primary" to get the main camera view
        primary_img_key = img_key_to_name["primary"]

        try:
            for i, episode in enumerate(dataset):
                # print progress
                print(
                    f"------------------- Processing episode {i+1} out of {min(dataset.cardinality().numpy(), n_samples)} of dataset {dataset_name} -------------------"
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
                            task = step["observation"][key].numpy().decode()
                            break
                        elif key in step:
                            task = step[key].numpy().decode()
                            break
                    # extract video
                    episode_images.append(step["observation"][primary_img_key].numpy())
                    # if the language instruction is None for some reason, skip the episode as it's weird
                    if task is None or task == "":
                        continue


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

                if task not in tasks_seen:
                    tasks_seen.add(task)
                    print(f"Tasks seen so far: {tasks_seen}")
                    f.create_group(task)
                    # TODO: get the lang embeddings for the task
                    task_embedding = embedding_text(model, tokenizer, [task]).detach().cpu().numpy()
                    # task_embedding = np.zeros((1024))  # TODO here
                    # create a dataset with the embeddings
                    f[task].create_dataset("lang_embedding", data=task_embedding)

                # get the task group
                task_group = f[task]
                # get its length
                task_group_len = len(task_group.keys())
                # convert length to string
                task_group_len_str = str(task_group_len)

                embedding_list = []
                for img in episode_images:
                    # center crop 224x224
                    image_embeddings = embedding_image(
                        model, processor, Image.fromarray(img.astype(np.uint8))
                    ).squeeze().detach().cpu().numpy()
                    embedding_list.append(image_embeddings)
                episode_image_embeddings = np.array(embedding_list) # TODO check dim, (n_frame, 1024)
                # TODO: get the embeddings for the images
                # episode_image_embeddings = np.zeros(
                #     (len(episode_images), 1024)
                # )  # TODO: here
                # create a dataset with the embeddings
                task_group.create_dataset(
                    task_group_len_str,
                    data=episode_image_embeddings,
                    compression="gzip",
                    compression_opts=9,
                )

                valid_samples_per_dataset += 1
                total_samples += 1
                print(
                    f"Valid total samples: {total_samples} "
                )
        except Exception as e:
            print(f"Failed to load dataset {dataset_name} with error: {e}")
            continue