import os
import h5py
import numpy as np
from tqdm import tqdm
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms
from reward_model import VLCRewardModel, RoboclipV2RewardModel
from reward_model.env_reward_model import EnvRewardModel


def lerobot_to_reward_hdf5(
    dataset_id,
    output_path,
    task_string,
    reward_model_type="roboclipv2",
    encoder_path=None,
    reward_model_path=None,
    device="cuda",
    batch_size=64,
    resolution=(640, 480),
    max_episodes=50,
    image_keys=["observation.images.main", "observation.images.side"],
    reward_image_key="observation.images.main",
):
    # Initialize the dataset
    if isinstance(dataset_id, str):
        dataset = LeRobotDataset(
            dataset_id,
            delta_timestamps=None,
            image_transforms=None,
            video_backend="pyav",
            local_files_only=True,
        )
    else:
        dataset = MultiLeRobotDataset(
            dataset_id,
            delta_timestamps=None,
            image_transforms=None,
            video_backend="pyav",
        )

    print("Saving to", output_path)

    # Initialize the reward model
    if reward_model_type == "roboclipv2":
        reward_model = RoboclipV2RewardModel(
            model_load_path=reward_model_path,
            use_pca=False,
            attention_heads=4,
            pca_model_dir=None,
            device=device,
            batch_size=batch_size,
        )
    elif reward_model_type == "vlc":
        reward_model = VLCRewardModel(
            encoder_path, device=device, batch_size=batch_size
        )
    elif reward_model_type in ["sparse", "dense"]:
        reward_model = EnvRewardModel(model_path=None)  # LIV encoder

    reward_image_idx = image_keys.index(reward_image_key)

    with h5py.File(output_path, "w") as h5_file:
        demo_number = 0
        prev_demo_number = 0

        states, next_states, actions, rewards, dones = [], [], [], [], []
        images_dict = {key: [] for key in image_keys}

        lang_embeddings = []
        policy_lang_embeddings = []

        for idx in tqdm(range(len(dataset)), total=len(dataset)):
            item = dataset[idx]
            demo_number = item["episode_index"].item()
            reward = 0  # Initial reward is 0
            done = False

            if (idx + 1) < len(dataset):
                next_item = dataset[idx + 1]
            else:
                break

            # Save trajectory data
            if demo_number != prev_demo_number:
                dones[-1] = True
                rewards[-1] = 1

                group = h5_file.create_group(str(demo_number))
                group["state"] = np.array(states)
                group["next_state"] = np.array(next_states)
                group["action"] = np.array(actions)
                group["reward"] = np.array(rewards)
                group["done"] = np.array(dones)

                for key in image_keys:
                    group[key] = np.array(images_dict[key])

                string_list = np.array([task_string] * len(states))
                group["string"] = string_list.astype("S")
                group["env_id"] = string_list.astype("S")

                # Reset for next demo
                states, next_states, actions, rewards, dones = [], [], [], [], []
                images_dict = {key: [] for key in image_keys}
                prev_demo_number = demo_number

            # Process rewards
            # instruction = item["string"].decode("utf-8")
            instruction = task_string
            text_embedding = reward_model.encode_text(instruction)[0]
            policy_lang_embedding = reward_model.encode_text_for_policy(instruction)[0]

            lang_embeddings.append(text_embedding)
            policy_lang_embeddings.append(policy_lang_embedding)

            states.append(item["observation.state"].numpy()[None, :])
            next_states.append(next_item["observation.state"].numpy()[None, :])
            actions.append(item["action"].numpy()[None, :])
            rewards.append(reward)
            dones.append(done)

            for key in image_keys:
                images_dict[key].append(item[key].numpy())

        # Save embeddings to HDF5
        h5_file.create_dataset(
            "lang_embedding", data=np.array(lang_embeddings), dtype="float32"
        )
        h5_file.create_dataset(
            "policy_lang_embedding",
            data=np.array(policy_lang_embeddings),
            dtype="float32",
        )


if __name__ == "__main__":
    dataset_id = "test/orange_left_right_handover"
    reward_model_type = "dense"
    output_path = f"./data/real_robot/updated_trajs/orange_left_right_handover_{reward_model_type}.h5"
    task_string = "pick the orange cup and move it to the right"
    lerobot_to_reward_hdf5(
        dataset_id=dataset_id,
        output_path=output_path,
        task_string=task_string,
        reward_model_type=reward_model_type,
        reward_model_path="/data/shared/roboclip/clip_liv_models/RegressionRandom_liv_subtract_before_heads_4/model_74.pt",
    )
