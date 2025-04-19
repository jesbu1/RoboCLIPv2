import argparse
import torch
import torch as th
import h5py
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward_model import VLCRewardModel, RoboclipV2RewardModel
from reward_model.env_reward_model import EnvRewardModel
# RoboCLIPEncoder


def compute_debug_reward(state):
    # In debug mode, we apply a manual reward function based on the current state
    state = state
    # # Let us set the task to be to approach a specific goal position
    goal_position = [
        90,
        0,
        0,
        0,
        0,
        0,
        90,
        0,
        0,
        0,
        0,
        0,
    ]

    goal_position = np.array(goal_position)

    # Reward is L2 distance to the goal position from state
    # reward = -torch.norm(state - goal_position)

    # The positions are rotations of motors, so we want the average degree difference
    difference = np.abs(state - goal_position)
    # Bound the difference to 180 degrees
    difference = np.minimum(difference, 180 - difference)

    reward = -np.sum(difference) / 12
    return reward


def label_trajectories_iteratively(
    args, traj_h5, output_file, image_keys, reward_image_key
):
    """
    Processes trajectories iteratively, computes rewards, and saves data directly to the output HDF5 file.
    If the output file already exists with embeddings, only updates the rewards.
    """
    # Check if this is just a reward update
    is_reward_update = all(
        key in output_file.keys() for key in ["lang_embedding", "img", "timesteps"]
    )

    # Initialize the specified encoder
    if args.reward_model == "roboclip":
        reward_model = RoboCLIPEncoder(
            args.encoder_path,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.reward_model == "vlc":
        reward_model = VLCRewardModel(
            args.encoder_path, device=args.device, batch_size=args.batch_size
        )
    elif args.reward_model == "roboclipv2":
        reward_model = RoboclipV2RewardModel(
            model_load_path=args.reward_model_path,
            use_pca=False,
            attention_heads=4,
            pca_model_dir=None,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.reward_model == "sparse":
        reward_model = EnvRewardModel(model_path=None)  # Uses a LIV encoder
    elif args.reward_model == "dense":
        reward_model = EnvRewardModel(model_path=None)  # Uses a LIV encoder
    elif args.reward_model == "debug":
        reward_model = EnvRewardModel(model_path=None)  # Uses a LIV encoder

    reward_image_idx = image_keys.index(reward_image_key)

    # If this is just a reward update, we can skip the embedding computation
    if is_reward_update:
        print("Output file exists with embeddings. Only updating rewards...")
        traj_keys = list(traj_h5.keys())
        total_timesteps = sum(len(traj_h5[traj_id]["reward"]) for traj_id in traj_keys)

        if "rewards" in output_file:
            del output_file["rewards"]  # Delete existing rewards
        rewards = output_file.create_dataset(
            "rewards", (total_timesteps,), dtype="float32"
        )

        current_timestep = 0
        for traj_id in tqdm(traj_keys, desc="Updating rewards"):
            traj_data = traj_h5[traj_id]
            num_steps = len(traj_data["done"])

            for i in range(num_steps):
                if not traj_data["done"][i]:
                    if args.reward_model == "dense":
                        rewards[current_timestep] = traj_data["reward"][i]
                    else:
                        rewards[current_timestep] = 0.0  # should be 0 right?
                else:
                    # Dense and sparse are special cases
                    if args.reward_model == "sparse":
                        rewards[current_timestep] = 1.0
                    elif args.reward_model == "dense":
                        rewards[current_timestep] = traj_data["reward"][i]
                    # Otherwise use the other reward models
                    else:
                        # Process video frames iteratively using stored embeddings
                        start_idx = max(0, i - args.window_length + 1)
                        video_embeddings = []
                        for j in range(start_idx, i + 1):
                            video_embeddings.append(
                                output_file[f"img_embedding_{reward_image_idx}"][
                                    current_timestep - (i - j)
                                ]
                            )
                        video_embedding = np.stack(video_embeddings)
                        text_embedding = output_file["lang_embedding"][current_timestep]

                        # Convert to torch tensors and reshape
                        video_embedding = torch.from_numpy(video_embedding)[None, ...]
                        text_embedding = (
                            torch.from_numpy(text_embedding)
                            .unsqueeze(0)
                            .repeat(1, video_embedding.shape[1], 1)
                        )
                        # Calculate reward
                        reward = reward_model.calculate_rewards(
                            text_embedding, video_embedding
                        )
                        rewards[current_timestep] = reward

                current_timestep += 1
        return

    # If we reach here, we need to do full processing
    # Initialize datasets in the output file
    traj_keys = list(traj_h5.keys())
    total_timesteps = sum(len(traj_h5[traj_id]["reward"]) for traj_id in traj_keys)
    output_file.create_dataset("rewards", (total_timesteps,), dtype="float32")
    output_file.create_dataset(
        "lang_embedding",
        (total_timesteps, reward_model.text_output_dim),
        dtype="float32",
    )
    output_file.create_dataset(
        "policy_lang_embedding",
        (total_timesteps, reward_model.policy_text_output_dim),
        dtype="float32",
    )

    for i, key in enumerate(image_keys):
        output_file.create_dataset(
            f"img_embedding_{i}",
            (total_timesteps, reward_model.img_output_dim),
            dtype="float32",
        )

    output_file.create_dataset("timesteps", (total_timesteps,), dtype="int32")

    # Determine image dataset shape and initialize it
    sample_img = traj_h5[traj_keys[0]][reward_image_key][
        0
    ]  # Sample image for shape and dtype
    # img_shape = (total_timesteps,) + sample_img.shape
    # img_dtype = sample_img.dtype
    # output_file.create_dataset("img", shape=img_shape, dtype=img_dtype)

    # image_datasets = {}
    # for key in image_keys:
    #     output_file.create_dataset(key, shape=img_shape, dtype=img_dtype)
    #     image_datasets[key] = output_file[key]

    rewards = output_file["rewards"]
    lang_embeds = output_file["lang_embedding"]
    policy_lang_embeds = output_file["policy_lang_embedding"]
    # img_embeds = output_file["img_embedding"]
    image_embeds_dict = {
        key: output_file[f"img_embedding_{i}"] for i, key in enumerate(image_keys)
    }

    timesteps = output_file["timesteps"]
    # img_dataset = output_file["img"]

    current_timestep = 0
    previous_instruction = None

    for traj_id in tqdm(traj_keys, desc="Processing trajectories"):
        traj_data = traj_h5[traj_id]
        num_steps = len(traj_data["done"])

        for i in tqdm(range(num_steps)):
            # Encode text only if the instruction changes
            if traj_data["string"][i] != previous_instruction:
                traj_string = traj_data["string"][i].decode("utf-8")
                text_embedding = reward_model.encode_text(traj_string)[0]

                policy_lang_embedding = reward_model.encode_text_for_policy(
                    traj_string
                )[0]

                assert len(text_embedding.shape) == 1
                previous_instruction = traj_string

            # Save language embedding and timestep
            lang_embeds[current_timestep] = text_embedding
            policy_lang_embeds[current_timestep] = policy_lang_embedding
            timesteps[current_timestep] = current_timestep

            # Use the image to get the image embedding
            for j, key in enumerate(image_keys):
                image = traj_data[key][i]
                image = image[None, None, :, :, :]
                image_embeds_dict[key][current_timestep] = reward_model.encode_images(
                    image
                ).squeeze()

            # img = traj_data["img"][i][None, None, ...]
            # img_embedding = reward_model.encode_images(img).squeeze()

            # img_embeds[current_timestep] = img_embedding

            # Compute reward
            if not traj_data["done"][i]:
                if args.reward_model == "dense":
                    rewards[current_timestep] = traj_data["reward"][i]
                elif args.reward_model == "debug":
                    # In debug mode, we apply a manual reward function based on the current state
                    state = traj_data["state"][i]
                    reward = compute_debug_reward(state)
                    rewards[current_timestep] = reward

                else:
                    rewards[current_timestep] = 0.0  # should be 0 right?

            else:
                # Dense and sparse are special cases
                if args.reward_model == "sparse":
                    rewards[current_timestep] = 1.0
                elif args.reward_model == "dense":
                    rewards[current_timestep] = traj_data["reward"][i]
                elif args.reward_model == "debug":
                    # In debug mode, we apply a manual reward function based on the current state
                    state = traj_data["state"][i]
                    reward = compute_debug_reward(state)
                    rewards[current_timestep] = reward

                # Otherwise use the other reward models
                else:
                    # Process video frames iteratively
                    # start_idx = max(0, i - args.window_length + 1)
                    # video_frames = [
                    #     traj_data[reward_image_key][j] for j in range(start_idx, i + 1)
                    # ]
                    # video_frames = np.stack(video_frames)[None, ...]
                    # video_embedding = reward_model.encode_images(video_frames)

                    # Should be of shape (1, num_frames, embedding_dim)
                    video_embeddings = image_embeds_dict[reward_image_key].unsqueeze(0)

                    # repeat the text embedding to match the batch size
                    text_embedding = (
                        torch.from_numpy(text_embedding)
                        .unsqueeze(0)
                        .repeat(1, video_embedding.shape[0], 1)
                    )
                    reward = reward_model.calculate_rewards(
                        text_embedding, torch.from_numpy(video_embedding)
                    )

                    rewards[current_timestep] = reward

            # Save the image for the current timestep
            # img_dataset[current_timestep] = traj_data["img"][i]
            # for key in image_keys:
            #     image_datasets[key][current_timestep] = traj_data[key][i]

            current_timestep += 1

    print(f"Successfully processed and saved {current_timestep} timesteps.")


def main():
    parser = argparse.ArgumentParser(description="Label rewards for trajectories.")
    parser.add_argument(
        "--trajs_to_label",
        required=True,
        help="Path to the trajectories file (HDF5 format).",
    )
    # parser.add_argument(
    #     "--output", required=True, help="Path to save the updated trajectories."
    # )
    parser.add_argument(
        "--reward_model",
        choices=["roboclipv2", "roboclip", "vlc", "dense", "sparse", "debug"],
        default="roboclipv2",
        help="Type of encoder to use.",
    )
    parser.add_argument("--encoder_path", help="Path to the encoder model file.")
    parser.add_argument(
        "--reward_model_path",
        help="Path to the saved model.",
        default="/data/shared/roboclip/clip_liv_models/RegressionRandom_liv_subtract_before_heads_4/model_74.pt",
    )
    parser.add_argument(
        "--sparse_only", action="store_true", help="Use sparse rewards only."
    )
    parser.add_argument(
        "--original_reward",
        action="store_true",
        help="Use original rewards if available.",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=1000000000000000000,
        help="Window length for video frame embeddings.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for encoding."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding video frames.",
    )

    # Todo: turn this into an argument

    # THIS IS FOR KOCH LEROBOT, which we moved to another script
    image_keys = ["observation.images.main", "observation.images.side"]
    image_keys = sorted(image_keys)
    reward_image_key = "observation.images.main"

    # This is for metaworld:
    image_keys = ["img"]
    reward_image_key = "img"

    args = parser.parse_args()

    # The path should be data/{path after data}/updated_trajs/{original file name}_{reward model}.h5
    path = f"data/{args.trajs_to_label.split('data/')[1].split('/')[0]}"
    # Create output file path in updated_traj folder
    # the last True indicates that it is a reward at every step
    output_path = f"{path}/updated_trajs/{os.path.basename(args.trajs_to_label[:-3])}_{args.reward_model}_True.h5"

    # Make sure directory exists
    if not os.path.exists(f"{path}/updated_trajs"):
        os.makedirs(f"{path}/updated_trajs")

    print(f"Saving to {output_path}")

    print("Loading trajectories...")
    with h5py.File(args.trajs_to_label, "r") as traj_file:
        if os.path.exists(output_path):
            # if False:
            print("Output file already exists. Updating rewards...")
            with h5py.File(output_path, "a") as output_file:
                label_trajectories_iteratively(
                    args, traj_file, output_file, image_keys, reward_image_key
                )
        else:
            with h5py.File(output_path, "w") as output_file:
                label_trajectories_iteratively(
                    args, traj_file, output_file, image_keys, reward_image_key
                )
                first_key = [key for key in traj_file.keys()][0]
                for key in traj_file[first_key].keys():
                    if key not in ["rewards"] + image_keys:
                        print(f"Saving {key}...")
                        items = []
                        for i in traj_file.keys():
                            items.extend(traj_file[str(i)][key])

                        try:
                            # breakpoint()
                            array_data = np.array(items)
                            output_file.create_dataset(
                                key, data=array_data, dtype=array_data.dtype
                            )
                        except:
                            print(f"Could not save {key}...")
                            breakpoint()

    print(f"Trajectories with rewards saved to {output_path}.")


if __name__ == "__main__":
    main()
