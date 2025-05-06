import os
import h5py
import numpy as np
from tqdm import tqdm
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms
from reward_model import VLCRewardModel, RoboclipV2RewardModel
from reward_model.env_reward_model import EnvRewardModel
from reward_model.rewind_reward_model import ReWiNDRewardModel
import glob
from usc_episode_rescaling import rescaling_dict


def lerobot_to_reward_hdf5(
    dataset_id,
    output_path,
    reward_model_type="roboclipv2",
    encoder_path=None,
    reward_model_path=None,
    device="cuda",
    batch_size=64,
    resolution=(640, 480),
    max_episodes=50,
    image_keys=["observation.images.main", "observation.images.side"],
    reward_image_key="observation.images.side",
    reward_at_every_step=False,
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
            local_files_only=True,
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
        reward_model = VLCRewardModel()
    elif reward_model_type in ["sparse", "dense"]:
        reward_model = EnvRewardModel(model_path=None)  # LIV encoder
    elif reward_model_type == "rewind":
        reward_model = ReWiNDRewardModel(
            model_load_path=reward_model_path,
            device=device,
            batch_size=batch_size,
            camera_names=image_keys,
        )

    reward_image_idx = image_keys.index(reward_image_key)

    # Get total size and shapes from first item
    total_size = len(dataset)
    sample_item = dataset[0]
    sample_task = dataset._datasets[0].meta.episodes[0]["tasks"][0]

    state_shape = sample_item["observation.state"].numpy()[None, :].shape[1:]
    action_shape = sample_item["action"].numpy()[None, :].shape[1:]

    # Get embedding dimensions from reward model
    policy_embedding_shape = reward_model.encode_text_for_policy(sample_task)[0].shape

    # Get image embedding shape by encoding a sample image
    sample_image = sample_item[image_keys[0]].numpy()[None, None, :, :, :]
    img_embedding_shape = (
        reward_model.encode_images_for_policy(sample_image).squeeze().shape
    )

    print(f"Total dataset size: {total_size}")
    print(f"State shape: {state_shape}")
    print(f"Action shape: {action_shape}")
    print(f"Policy embedding shape: {policy_embedding_shape}")
    print(f"Image embedding shape: {img_embedding_shape}")

    try:
        text_embedding_shape = reward_model.encode_text(sample_task)[0].shape
        print(f"Text embedding shape: {text_embedding_shape}")
    except Exception as e:
        text_embedding_shape = None

    with h5py.File(output_path, "w") as h5_file:
        # Create fixed-size datasets with chunks and maxshape for resizing
        states_dataset = h5_file.create_dataset(
            "state",
            shape=(total_size, *state_shape),
            maxshape=(None, *state_shape),
            chunks=True,
            dtype=np.float32,
        )
        actions_dataset = h5_file.create_dataset(
            "action",
            shape=(total_size, *action_shape),
            maxshape=(None, *action_shape),
            chunks=True,
            dtype=np.float32,
        )
        rewards_dataset = h5_file.create_dataset(
            "rewards",
            shape=(total_size,),
            maxshape=(None,),
            chunks=True,
            dtype=np.float32,
        )
        dones_dataset = h5_file.create_dataset(
            "done", shape=(total_size,), maxshape=(None,), chunks=True, dtype=np.bool_
        )

        # Create embedding datasets for each image source
        image_embeds_datasets = {}
        for i, key in enumerate(image_keys):
            image_embeds_datasets[key] = h5_file.create_dataset(
                f"img_embedding_{i}",
                shape=(total_size, *img_embedding_shape),
                maxshape=(None, *img_embedding_shape),
                chunks=True,
                dtype=np.float32,
            )

        if text_embedding_shape is not None:
            # Create embedding datasets
            lang_embedding_dataset = h5_file.create_dataset(
                "lang_embedding",
                shape=(total_size, *text_embedding_shape),
                maxshape=(None, *text_embedding_shape),
                chunks=True,
                dtype=np.float32,
            )
        else:
            # Then we store only the string
            lang_embedding_dataset = h5_file.create_dataset(
                "lang_embedding",
                shape=(total_size,),
                maxshape=(None,),
                chunks=True,
                dtype=h5py.string_dtype(),
            )

        policy_lang_embedding_dataset = h5_file.create_dataset(
            "policy_lang_embedding",
            shape=(total_size, *policy_embedding_shape),
            maxshape=(None, *policy_embedding_shape),
            chunks=True,
            dtype=np.float32,
        )

        # Create string dataset for task descriptions
        string_dataset = h5_file.create_dataset(
            "string",
            shape=(total_size,),
            maxshape=(None,),
            chunks=True,
            dtype=h5py.string_dtype(),
        )
        env_id_dataset = h5_file.create_dataset(
            "env_id",
            shape=(total_size,),
            maxshape=(None,),
            chunks=True,
            dtype=h5py.string_dtype(),
        )

        # Process episodes
        current_idx = 0
        prev_task = None
        episode_start_idx = 0
        episode_items = []

        prev_episode_idx = None
        episode_index = None

        for idx in tqdm(range(total_size)):
            item = dataset[idx]

            # Get task string
            episode_index = item["episode_index"]

            task = dataset._datasets[item["dataset_index"]].meta.episodes[
                item["episode_index"]
            ]["tasks"][0]
            # capitalize the first letter of the task
            task = task.capitalize()
            if prev_task is None:
                prev_task = task

            # Check if we're at a new task/episode boundary
            new_episode = (
                (idx > 0 and task != prev_task)
                or (idx == total_size - 1)
                or (episode_index != prev_episode_idx)
            )

            if new_episode and episode_items:
                # This is because the current task item is the next one
                print(f"New episode, {prev_task}")
                # Process the completed episode
                episode_len = len(episode_items)

                # Apply rescaling based on dataset ID
                repo_id = (
                    dataset_id
                    if isinstance(dataset_id, str)
                    else dataset_id[item["dataset_index"]]
                )
                if repo_id in rescaling_dict:
                    rescaling_factor = rescaling_dict[repo_id]
                    keep_frames = int(episode_len * rescaling_factor)

                    # Rescale frames
                    print(
                        f"Rescaling episode {episode_index} from {episode_len} to {keep_frames}"
                    )
                    episode_items = episode_items[:keep_frames]
                    print(f"New episode length: {len(episode_items)}")
                else:
                    print(f"No rescaling for {repo_id}")

                # Sample frames uniformly
                # if keep_frames < episode_len:
                #     indices = np.linspace(0, episode_len - 1, keep_frames, dtype=int)
                #     episode_items = [episode_items[i] for i in indices]

                image_embeddings = []
                rewards = []
                # Process the episode items
                for ep_idx, ep_item in enumerate(episode_items):
                    # Only compute text embeddings once per episode
                    if ep_idx == 0:
                        text_embedding = reward_model.encode_text(prev_task)[0]
                        policy_lang_embedding = reward_model.encode_text_for_policy(
                            prev_task
                        )[0]

                    # Write to datasets
                    states_dataset[current_idx] = ep_item["observation.state"].numpy()[
                        None, :
                    ][0]
                    actions_dataset[current_idx] = ep_item["action"].numpy()[None, :][0]
                    dones_dataset[current_idx] = False

                    # Compute and write image embeddings
                    timestep_image_embeddings = []
                    for key in image_keys:
                        image = ep_item[key].numpy()[None, None, :, :, :]
                        image_embedding = reward_model.encode_images_for_policy(
                            image
                        ).squeeze()
                        image_embeds_datasets[key][current_idx] = image_embedding

                        reward_image_embedding = reward_model.encode_images(
                            image
                        ).squeeze()

                        timestep_image_embeddings.append(reward_image_embedding)
                    image_embeddings.append(timestep_image_embeddings)

                    # Write embeddings and strings
                    lang_embedding_dataset[current_idx] = text_embedding
                    policy_lang_embedding_dataset[current_idx] = policy_lang_embedding
                    string_dataset[current_idx] = prev_task.encode("utf-8")
                    env_id_dataset[current_idx] = prev_task.encode("utf-8")

                    if not reward_at_every_step or ep_idx == 0:
                        rewards_dataset[current_idx] = 0
                    else:
                        embeddings = np.array(image_embeddings)
                        # Compute the rewards
                        sum_rewards = 0
                        for i, image_key in enumerate(image_keys):
                            if isinstance(text_embedding, np.ndarray):
                                text_embedding = text_embedding[None, None, :]
                            sum_rewards += reward_model.calculate_rewards(
                                text_embedding,
                                embeddings[None, :, i],
                                image_key,
                            )

                        sum_rewards /= len(image_keys)
                        rewards_dataset[current_idx] = sum_rewards
                        rewards.append(sum_rewards)

                    current_idx += 1

                # # Set reward and done for the last frame of the episode
                if current_idx > 0:
                    # rewards_dataset[current_idx - 1] = 1
                    dones_dataset[current_idx - 1] = True

                    image_embeddings = np.array(image_embeddings)
                    # Compute the rewards
                    sum_rewards = 0
                    for i, image_key in enumerate(image_keys):
                        if isinstance(text_embedding, np.ndarray):
                            text_embedding = text_embedding[None, None, :]
                        sum_rewards += reward_model.calculate_rewards(
                            text_embedding,
                            image_embeddings[None, :, i],
                            image_key,
                        )
                    sum_rewards /= len(image_keys)
                    reward = sum_rewards
                    rewards_dataset[current_idx] = reward
                    print(
                        f"Reward: {np.array(rewards).round(3).reshape(-1).tolist()} for task: {task}"
                    )

                # Reset for next episode
                episode_items = [item]
            else:
                episode_items.append(item)

            prev_task = task
            prev_episode_idx = episode_index

        # Update total size to actual number of frames saved
        if current_idx < total_size:
            states_dataset.resize((current_idx, *state_shape))
            actions_dataset.resize((current_idx, *action_shape))
            rewards_dataset.resize((current_idx,))
            dones_dataset.resize((current_idx,))
            if text_embedding_shape is not None:
                lang_embedding_dataset.resize((current_idx, *text_embedding_shape))
            else:
                lang_embedding_dataset.resize((current_idx,))
            policy_lang_embedding_dataset.resize((current_idx, *policy_embedding_shape))
            string_dataset.resize((current_idx,))
            env_id_dataset.resize((current_idx,))
            for key in image_keys:
                image_embeds_datasets[key].resize((current_idx, *img_embedding_shape))


if __name__ == "__main__":
    path = "/home/abrar/.cache/huggingface/lerobot/usc_koch_rewind/"
    dataset_ids = glob.glob(path + "/*")
    dataset_ids = [f"usc_koch_rewind/{os.path.basename(x)}" for x in dataset_ids]

    print(dataset_ids)

    eval_tasks = [
        "usc_koch_rewind/put_the_blue_cup_on_the_red_plate",
        "usc_koch_rewind/separate_the_orange_and_blue_cups",
        "usc_koch_rewind/open_the_red_trash_bin",
        "usc_koch_rewind/throw_the_banana_away_in_the_red_trash_bin",
        "usc_koch_rewind/put_the_red_tape_in_the_box_on_the_right",
    ]

    # /home/abrar/.cache/huggingface/lerobot/usc_koch_rewind/Scrub_the_yellow_plate_with_the_sponge_2
    # dataset_ids = ["usc_koch_rewind/Scrub_the_blue_plate_with_the_yellow_sponge_2"]

    # remove eval tasks from dataset_ids
    # dataset_ids = [x for x in dataset_ids if x not in eval_tasks]

    # remove anything without a _2 on it
    # dataset_ids = [x for x in dataset_ids if "_2" in x]

    # reward_model_path = "weights/rewind/one_step_transformer.pth"
    # reward_model_path = ["weights/rewind/all/model_49.pth"] * 2
    reward_model_path = ["weights/rewind/new_data_mix/model_99.pth"] * 2

    # reward_model_path = "weights/rewind/real_world_PosEmb_Rewind_ratio_0.8_EMA_momentum_0.3_End_Rewind_ratio_0.1/model_30.pth"

    reward_image_key = "observation.images.main"

    reward_at_every_step = True
    # dataset_id = "test/orange_left_right_handover"
    reward_model_type = "vlc"
    # output_path = f"./data/real_robot/updated_trajs/usc_koch_rewind_new_data_only_new_reward_{reward_model_type}_{reward_at_every_step}.h5"

    output_path = f"./data/real_robot/updated_trajs/usc_koch_rewind_new_old_data_new_reward_{reward_model_type}_{reward_at_every_step}.h5"
    lerobot_to_reward_hdf5(
        dataset_id=dataset_ids,
        output_path=output_path,
        reward_model_type=reward_model_type,
        reward_model_path=reward_model_path,
        reward_at_every_step=reward_at_every_step,
        reward_image_key=reward_image_key,
    )
