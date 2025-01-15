import h5py
import os
import numpy as np
from tqdm import tqdm


import logging

import torch
from omegaconf import ListConfig, OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms

from lerobot.common.datasets.sampler import EpisodeAwareSampler


def convert_lerobot_to_hdf5(
    dataset_id, output_path, task_string, resolution=(640, 480), max_episodes=50
):
    # get lerobot data
    if isinstance(dataset_id, str):
        # TODO (aliberts): add 'episodes' arg from config after removing hydra

        image_transforms = None

        delta_timestamps = None
        video_backend = "pyav"

        dataset = LeRobotDataset(
            dataset_id,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=video_backend,
            local_files_only=True,
        )
    else:
        dataset = MultiLeRobotDataset(
            dataset_id,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=video_backend,
        )

    print("Saving to", output_path)

    with h5py.File(output_path, "w") as h5_file:
        demo_number = 0
        prev_demo_number = 0

        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        images = []
        env_ids = []

        # NOTE: We are assuming that dataset is sequentially increasing
        for idx in tqdm(range(len(dataset)), total=len(dataset)):
            item = dataset[idx]

            demo_number = item["episode_index"].item()
            reward = 0  # All 0 reward
            done = False  # Not done until the end

            if (idx + 1) < len(dataset):
                next_item = dataset[idx + 1]
            else:
                break

            # Save
            if demo_number != prev_demo_number:
                # Then we turn the last item into a done
                dones[-1] = True
                rewards[-1] = 1

                print(demo_number)

                group = h5_file.create_group(str(demo_number))
                group["state"] = np.array(states)
                group["next_state"] = np.array(next_states)
                group["action"] = np.array(actions)
                group["reward"] = np.array(rewards)
                group["done"] = np.array(dones)
                group["img"] = np.array(images)

                string_list = np.array([task_string] * len(states))

                group["string"] = np.array(string_list).astype("S")  # maybe used
                group["env_id"] = np.array(string_list).astype("S")  # not used

                states = []
                next_states = []
                actions = []
                rewards = []
                dones = []
                images = []
                env_ids = []

                prev_demo_number = demo_number
                demo_number += 1

            states.extend(item["observation.state"].numpy()[None, :])
            next_states.extend(next_item["observation.state"].numpy()[None, :])
            actions.extend(item["action"].numpy()[None, :])
            rewards.extend([reward])
            dones.extend([done])

            images.append(item["observation.images.main"].numpy())


if __name__ == "__main__":
    # dataset_id = "test/pick_all"
    # output_path = "./data/real_robot/orig/pick_all.h5"
    # task_string = "pick_all"

    dataset_id = "test/pick_orange_left_right"
    output_path = "./data/real_robot/orig/pick_orange_left_right.h5"
    task_string = "pick the orange cup and move it to the right"
    convert_lerobot_to_hdf5(
        dataset_id=dataset_id,
        output_path=output_path,
        task_string=task_string,
        resolution=(640, 480),
        max_episodes=50,
    )
