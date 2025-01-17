import tensorflow_datasets as tfds
from tqdm import tqdm
import os
from oxe_configs import OXE_DATASET_CONFIGS
import json
import numpy as np
import h5py

TFDS_PATH = ""
SAVE_H5_NAME = ""  # name of the h5 file it'll be saved to
DEBUG = True  # will only make 5
SPECIFIC_TASKS = "bridge,kuka"

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "natural_language_instruction",
    "language_instruction",
]


DATASET_TRANSFORMS = (
    # Datasets used for OpenVLA: https://openvla.github.io/
    "fractal20220817_data 0.1.0 resize_and_jpeg_encode",
    "bridge 0.1.0 resize_and_jpeg_encode",
    "kuka 0.1.0 resize_and_jpeg_encode,filter_success",
    "taco_play 0.1.0 resize_and_jpeg_encode",
    "jaco_play 0.1.0 resize_and_jpeg_encode",
    "berkeley_cable_routing 0.1.0 resize_and_jpeg_encode",
    "roboturk 0.1.0 resize_and_jpeg_encode",
    "viola 0.1.0 resize_and_jpeg_encode",
    "berkeley_autolab_ur5 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels",
    "toto 0.1.0 resize_and_jpeg_encode",
    "language_table 0.1.0 resize_and_jpeg_encode",
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels",
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "bc_z 0.1.0 resize_and_jpeg_encode",
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "utaustin_mutex 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels",
    "berkeley_fanuc_manipulation 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels",
    "cmu_stretch 0.1.0 resize_and_jpeg_encode",
    "dobbe 0.0.1 resize_and_jpeg_encode",
    "fmb 0.0.1 resize_and_jpeg_encode",
    "droid 1.0.0 resize_and_jpeg_encode",
)

dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]

if SPECIFIC_TASKS is not None:
    # overwrite the dataset_names with the specific tasks
    dataset_names = SPECIFIC_TASKS.split(",")

# make a set to keep track of the tasks we've seen
tasks_seen = set()

with h5py.File(SAVE_H5_NAME, "w") as f:
    for dataset_name in tqdm(dataset_names):
        dataset = tfds.load(dataset_name, data_dir=TFDS_PATH, split="train")
        n_samples = 5 if DEBUG else 1000000000000000000
        valid_samples = 0
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name][
            "image_obs_keys"
        ]  # dict mapping img_keys to the names of the images in OXE

        # get the image key that matches "primary" to get the main camera view
        primary_img_key = img_key_to_name["primary"]

        for i, episode in enumerate(dataset):
            # print progress
            print(
                f"------------------- Processing episode {i+1} out of {min(dataset.cardinality().numpy(), n_samples)} of dataset {dataset_name} -------------------"
            )
            # skip if we have already saved the video
            this_episode_name = f"{dataset_name}_ep{i}"

            episode_images = []

            if valid_samples >= n_samples:
                break
            # task is the language instruction
            task = None
            for _, step in enumerate(episode["steps"]):
                # skip data loading if no lang
                for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
                    if key in step["observation"]:
                        task = step["observation"][key].numpy().decode()
                    # if the language instruction is None for some reason, skip the episode as it's weird
                    if task is None or task == "":
                        print(
                            f"Skipping episode {i} of dataset {dataset_name} as the task is None or empty."
                        )
                        break

                # extract video
                episode_images.append(step["observation"][primary_img_key].numpy())

            # process task name to capitalize the first letter
            task = task.capitalize()

            if task not in tasks_seen:
                tasks_seen.add(task)
                print(f"Tasks seen so far: {tasks_seen}")
                f.create_group(task)
                # TODO: get the lang embeddings for the task
                task_embedding = np.zeros((1024))  # TODO here
                # create a dataset with the embeddings
                f[task].create_dataset("embedding", data=task_embedding)

            # get the task group
            task_group = f[task]
            # get its length
            task_group_len = len(task_group.keys())
            # convert length to string
            task_group_len_str = str(task_group_len)
            # TODO: get the embeddings for the images
            episode_image_embeddings = np.zeros(
                (len(episode_images), 1024)
            )  # TODO: here
            # create a dataset with the embeddings
            task_group.create_dataset(
                task_group_len_str,
                data=episode_image_embeddings,
                compression="gzip",
                compression_opts=5,
            )

            valid_samples += 1
