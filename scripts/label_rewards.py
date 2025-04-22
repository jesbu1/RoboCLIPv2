import argparse
import torch
import torch as th
import h5py
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.reward_model import VLCRewardModel, RoboclipV2RewardModel
from models.reward_model.env_reward_model import EnvRewardModel
# RoboCLIPEncoder

def label_trajectories_iteratively(args, traj_h5, output_file):
    """
    Processes trajectories iteratively, computes rewards, and saves data directly to the output HDF5 file.
    If the output file already exists with embeddings, only updates the rewards.
    """
    # Check if this is just a reward update
    is_reward_update = all(key in output_file.keys() for key in ['img_embedding', 'lang_embedding', 'img', 'timesteps'])
    
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
        reward_model = EnvRewardModel(model_path=None) # Uses a LIV encoder
    elif args.reward_model == "dense":
        reward_model = EnvRewardModel(model_path=None) # Uses a LIV encoder

    # If this is just a reward update, we can skip the embedding computation
    if is_reward_update:
        print("Output file exists with embeddings. Only updating rewards...")
        traj_keys = list(traj_h5.keys())
        total_timesteps = sum(len(traj_h5[traj_id]["reward"]) for traj_id in traj_keys)
        
        if "rewards" in output_file:
            del output_file["rewards"]  # Delete existing rewards
        rewards = output_file.create_dataset("rewards", (total_timesteps,), dtype="float32")
        
        current_timestep = 0
        for traj_id in tqdm(traj_keys, desc="Updating rewards"):
            traj_data = traj_h5[traj_id]
            num_steps = len(traj_data["done"])
            
            for i in range(num_steps):
                if not traj_data["done"][i]:
                    if args.reward_model == "dense":
                        rewards[current_timestep] = traj_data["reward"][i]
                    else:
                        rewards[current_timestep] = 1.0
                else:
                    # Dense and sparse are special cases
                    if args.reward_model == "sparse":
                        rewards[current_timestep] = 1.0
                    elif args.reward_model == "dense":
                        rewards[current_timestep] = traj_data["reward"][i]
                    # Otherwise use the other reward models
                    else:
                        # Process video frames iteratively using stored embeddings
                        # TODO: fix this for rewind inference
                        start_idx = max(0, i - args.window_length + 1)
                        video_embeddings = []
                        for j in range(start_idx, i + 1):
                            video_embeddings.append(output_file["img_embedding"][current_timestep - (i - j)])
                        video_embedding = np.stack(video_embeddings)
                        text_embedding = output_file["lang_embedding"][current_timestep]
                        
                        # Convert to torch tensors and reshape
                        video_embedding = torch.from_numpy(video_embedding)[None, ...]
                        text_embedding = torch.from_numpy(text_embedding).unsqueeze(0).repeat(1, video_embedding.shape[1], 1)
                        # Calculate reward
                        reward = reward_model.calculate_rewards(text_embedding, video_embedding)
                        rewards[current_timestep] = reward

                current_timestep += 1
        return

    # If we reach here, we need to do full processing
    # Initialize datasets in the output file
    traj_keys = list(traj_h5.keys())
    total_timesteps = sum(len(traj_h5[traj_id]["reward"]) for traj_id in traj_keys)
    output_file.create_dataset("rewards", (total_timesteps,), dtype="float32")
    output_file.create_dataset(
        "lang_embedding", (total_timesteps, reward_model.text_output_dim), dtype="float32"
    )
    output_file.create_dataset(
        "policy_lang_embedding", (total_timesteps, reward_model.policy_text_output_dim), dtype="float32"
    )
    output_file.create_dataset(
        "img_embedding", (total_timesteps, reward_model.img_output_dim), dtype="float32"
    )

    output_file.create_dataset("timesteps", (total_timesteps,), dtype="int32")

    # Determine image dataset shape and initialize it
    sample_img = traj_h5[traj_keys[0]]["img"][0]  # Sample image for shape and dtype
    img_shape = (total_timesteps,) + sample_img.shape
    img_dtype = sample_img.dtype
    output_file.create_dataset("img", shape=img_shape, dtype=img_dtype)

    rewards = output_file["rewards"]
    lang_embeds = output_file["lang_embedding"]
    policy_lang_embeds = output_file["policy_lang_embedding"]
    img_embeds = output_file["img_embedding"]
    timesteps = output_file["timesteps"]
    img_dataset = output_file["img"]

    current_timestep = 0
    previous_instruction = None

    for traj_id in tqdm(traj_keys, desc="Processing trajectories"):
        traj_data = traj_h5[traj_id]
        num_steps = len(traj_data["done"])

        for i in range(num_steps):
            # Encode text only if the instruction changes
            if traj_data["string"][i] != previous_instruction:
                traj_string = traj_data["string"][i].decode("utf-8")
                text_embedding = reward_model.encode_text(traj_string)[0]

                policy_lang_embedding = reward_model.encode_text_for_policy(traj_string)[0]

                assert len(text_embedding.shape) == 1
                previous_instruction = traj_string

            # Save language embedding and timestep
            lang_embeds[current_timestep] = text_embedding
            policy_lang_embeds[current_timestep] = policy_lang_embedding
            timesteps[current_timestep] = current_timestep

            # Use the image to get the image embedding
            img = traj_data["img"][i][None, None,...]
            img_embedding = reward_model.encode_images(img).squeeze()

            img_embeds[current_timestep] = img_embedding

            # Compute reward
            if not traj_data["done"][i]:
                if args.reward_model == "dense":
                    rewards[current_timestep] = traj_data["reward"][i]
                else:
                    rewards[current_timestep] = 1.0

            else:
                # Dense and sparse are special cases
                if args.reward_model == "sparse":
                    rewards[current_timestep] = 1.0
                elif args.reward_model == "dense":
                    rewards[current_timestep] = traj_data["reward"][i]

                # Otherwise use the other reward models
                else:
                    # Process video frames iteratively
                    start_idx = max(0, i - args.window_length + 1)
                    video_frames = [
                        traj_data["img"][j] for j in range(start_idx, i + 1)
                    ]
                    video_frames = np.stack(video_frames)[None, ...]
                    video_embedding = reward_model.encode_images(video_frames)
                    # repeat the text embedding to match the batch size
                    text_embedding = torch.from_numpy(text_embedding).unsqueeze(0).repeat(1, video_embedding.shape[0], 1)
                    reward = reward_model.calculate_rewards(text_embedding, torch.from_numpy(video_embedding))

                    rewards[current_timestep] = reward

            # Save the image for the current timestep
            img_dataset[current_timestep] = traj_data["img"][i]

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
        choices=["roboclipv2", "roboclip", "vlc", "dense", "sparse"],
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

    args = parser.parse_args()

    # Create output file path in updated_traj folder
    output_path = f"data/h5_buffers/updated_trajs/{os.path.basename(args.trajs_to_label[:-3])}_{args.reward_model}.h5"

    print(f"Saving to {output_path}")

    print("Loading trajectories...")
    with h5py.File(args.trajs_to_label, "r") as traj_file:
        if os.path.exists(output_path):
            print("Output file already exists. Updating rewards...")
            with h5py.File(output_path, "a") as output_file:
                label_trajectories_iteratively(args, traj_file, output_file)
        else:
            with h5py.File(output_path, "w") as output_file:
                label_trajectories_iteratively(args, traj_file, output_file)

                for key in traj_file["0"].keys():
                    if key not in ["rewards", "img"]:
                        print(f"Saving {key}...")
                        items = []
                        for i in range(len(traj_file.keys())):
                            items.extend(traj_file[str(i)][key])

                        try:
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
