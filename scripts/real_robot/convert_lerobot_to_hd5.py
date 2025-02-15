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

    image_transforms = None
    delta_timestamps = None
    video_backend = "pyav"
    if isinstance(dataset_id, str):
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
            local_files_only=True,
        )

    print("Saving to", output_path)

    image_keys = ["observation.images.main", "observation.images.side"]

    with h5py.File(output_path, "w") as h5_file:
        demo_number = 0
        prev_demo_number = 0

        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        # images = []
        images_dict = {}
        for key in image_keys:
            images_dict[key] = []

        env_ids = []

        episode_count = 0

        # NOTE: We are assuming that dataset is sequentially increasing
        for idx in tqdm(range(len(dataset)), total=len(dataset)):
            item = dataset[idx]

            if task_string is None:
                task_string = dataset._datasets[item["dataset_index"]].meta.episodes[
                    item["episode_index"]
                ]["tasks"][0]

            # demo_number = item["episode_index"].item() * item['dataset_index'].item()
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

                group = h5_file.create_group(str(episode_count))
                episode_count += 1

                group["state"] = np.array(states)
                group["next_state"] = np.array(next_states)
                group["action"] = np.array(actions)
                group["reward"] = np.array(rewards)
                group["done"] = np.array(dones)
                # group["img"] = np.array(images)

                for key in image_keys:
                    group[key] = np.array(images_dict[key])

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

                for key in image_keys:
                    images_dict[key] = []

                prev_demo_number = demo_number
                demo_number += 1

            states.extend(item["observation.state"].numpy()[None, :])
            next_states.extend(next_item["observation.state"].numpy()[None, :])
            actions.extend(item["action"].numpy()[None, :])
            rewards.extend([reward])
            dones.extend([done])

            # images.append(item["observation.images.main"].numpy())

            for key in image_keys:
                # Need unormalize
                images_dict[key].append((item[key] * 255).numpy())


if __name__ == "__main__":
    # dataset_id = "test/pick_all"
    # output_path = "./data/real_robot/orig/pick_all.h5"
    # task_string = "pick_all"

    import glob

    path = "/home/abrar/.cache/huggingface/lerobot/usc_koch_rewind/"
    dataset_ids = glob.glob(path + "/*")

    dataset_ids = [f"usc_koch_rewind/{os.path.basename(x)}" for x in dataset_ids]
    task_string = None

    # dataset_id = "test/orange_left_right_handover"
    # output_path = "./data/real_robot/orig/orange_left_right_handover.h5"
    output_path = "./data/real_robot/orig/usc_koch_rewind.h5"
    # task_string = "pick the orange cup and move it to the right"
    convert_lerobot_to_hdf5(
        dataset_id=dataset_ids,
        output_path=output_path,
        task_string=task_string,
        resolution=(640, 480),
        max_episodes=50,
    )
