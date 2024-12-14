import argparse
import torch
import torch as th
import h5py
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from encoders import XCLIPEncoder, S3DEncoder, VLCEncoder  # Assume these are imported or defined elsewhere

from transformations import LinearTransform, PCATransform, NoTransform
class RewardLabeler:
    def __init__(self, args):
        self.args = args

        # Initialize the specified encoder
        if args.encoder_type == "xclip":
            self.encoder = XCLIPEncoder()
        elif args.encoder_type == "s3d":
            self.encoder = S3DEncoder(args.encoder_path)
        elif args.encoder_type == "vlc":
            self.encoder = VLCEncoder(args.encoder_path)

        # Load the transformation model
        if args.transform_model_path:

            if transform_model == "linear":
                self.transform_model = LinearTransform(args)
            elif transform_model == "pca":
                self.transform_model = PCATransform(args)
            else:
                self.transform_model = NoTransform(args)

            self.transform_model = SingleLayerMLP(512, 512, normalize=True)
            dict = torch.load(args.transform_model_path)
            self.transform_model.load_state_dict(dict['model_state_dict'])
            self.transform_model.eval().cuda()

    def label_trajectories(self, args, traj_h5):
        """
        Densely labels rewards for each trajectory based on encoded video frames.
        :param traj_data: Trajectory data containing video frames to be labeled.
        :return: Rewards for each trajectory.
        """
        all_rewards = []
        all_lang_embeds = []
        timesteps = []

        # Let's reorganize the traj_h5 into traj_data to be flat
        # Currently it should be traj_data['0']['done'], etc/
        # Let's make it traj_data['done'], etc.
        # TODO: This is inefficient, but it's fine for now.
        traj_data = {}
        for i in traj_h5.keys():
            for key in traj_h5[i].keys():
                if key not in traj_data.keys():
                    traj_data[key] = []
                traj_data[key].extend(traj_h5[i][key])

        previous_instruction = None
        # import pdb; pdb.set_trace()
        for i in tqdm(range(len(traj_data['done']))):
            if traj_data['string'][i] != previous_instruction:
                text_embedding = self.encoder.encode_text(traj_data['string'][i])
                previous_instruction = traj_data['string'][i]

            all_lang_embeds.append(text_embedding.cpu().numpy())
            timesteps.append(i)

            if not traj_data['done'][i]:
                if args.original_reward:
                    all_rewards.append(traj_data['reward'][i])
                else:
                    all_rewards.append(0.0)
            else:
                if args.sparse_only:
                    all_rewards.append(1.0)
                elif args.original_reward:
                    all_rewards.append(traj_data['reward'][i])
                else:
                    video_frames = traj_data['img'][timesteps[-args.window_length:]]
                    video_embedding = self.encoder.encode_video(video_frames)

                    if hasattr(self, 'transform_model'):
                        video_embedding = self.transform_model.apply_transform(video_embedding)

                    similarity = self.compute_similarity(video_embedding, text_embedding)
                    all_rewards.append(similarity)

        return all_rewards, all_lang_embeds, timesteps, traj_data

    def compute_similarity(self, video_embedding, text_embedding):
        """
        Computes similarity between video embedding and a precomputed text embedding.
        :param video_embedding: Encoded video frames.
        :return: Similarity score as reward.
        """
        similarity = torch.matmul(text_embedding, video_embedding.t()).cpu().numpy()[0][0]
        return similarity
def label_trajectories_iteratively(args, traj_h5, output_file):
    """
    Processes trajectories iteratively, computes rewards, and saves data directly to the output HDF5 file.
    """
    reward_labeler = RewardLabeler(args)

    # Initialize datasets in the output file
    traj_keys = list(traj_h5.keys())
    total_timesteps = sum(len(traj_h5[traj_id]['reward']) for traj_id in traj_keys)
    output_file.create_dataset('rewards', (total_timesteps,), dtype='float32')
    output_file.create_dataset('lang_embedding', (total_timesteps, 512), dtype='float32')  # Assuming 512 for embedding
    output_file.create_dataset('timesteps', (total_timesteps,), dtype='int32')

    # Determine image dataset shape and initialize it
    sample_img = traj_h5[traj_keys[0]]['img'][0]  # Sample image for shape and dtype
    img_shape = (total_timesteps,) + sample_img.shape
    img_dtype = sample_img.dtype
    output_file.create_dataset('img', shape=img_shape, dtype=img_dtype)

    rewards = output_file['rewards']
    lang_embeds = output_file['lang_embedding']
    timesteps = output_file['timesteps']
    img_dataset = output_file['img']

    current_timestep = 0
    previous_instruction = None

    for traj_id in tqdm(traj_keys, desc="Processing trajectories"):
        traj_data = traj_h5[traj_id]
        num_steps = len(traj_data['done'])

        for i in range(num_steps):
            # Encode text only if the instruction changes
            if traj_data['string'][i] != previous_instruction:
                text_embedding = reward_labeler.encoder.encode_text(traj_data['string'][i])
                previous_instruction = traj_data['string'][i]

            # Save language embedding and timestep
            lang_embeds[current_timestep] = text_embedding.cpu().numpy()
            timesteps[current_timestep] = current_timestep

            # Compute reward
            if not traj_data['done'][i]:
                rewards[current_timestep] = traj_data['reward'][i] if args.original_reward else 0.0
            else:
                if args.sparse_only:
                    rewards[current_timestep] = 1.0
                elif args.original_reward:
                    rewards[current_timestep] = traj_data['reward'][i]
                else:
                    # Process video frames iteratively
                    start_idx = max(0, i - args.window_length + 1)
                    video_frames = [traj_data['img'][j] for j in range(start_idx, i + 1)]
                    video_embedding = reward_labeler.encoder.encode_video(video_frames)

                    if hasattr(reward_labeler, 'transform_model'):
                        video_embedding = reward_labeler.transform_model.apply_transform(video_embedding)

                    similarity = reward_labeler.compute_similarity(video_embedding, text_embedding)
                    rewards[current_timestep] = similarity

            # Save the image for the current timestep
            img_dataset[current_timestep] = traj_data['img'][i]

            current_timestep += 1

    print(f"Successfully processed and saved {current_timestep} timesteps.")


def main():
    parser = argparse.ArgumentParser(description="Label rewards for trajectories.")
    parser.add_argument('--trajs_to_label', required=True, help='Path to the trajectories file (HDF5 format).')
    parser.add_argument('--output', required=True, help='Path to save the updated trajectories.')
    parser.add_argument('--encoder_type', choices=['xclip', 's3d', 'vlc'], default='xclip', help='Type of encoder to use.')
    parser.add_argument('--encoder_path', help='Path to the encoder model file.')
    parser.add_argument('--transform_model_path', help='Path to the transformation model file.', default=None)
    parser.add_argument('--sparse_only', action="store_true", help='Use sparse rewards only.')
    parser.add_argument('--original_reward', action="store_true", help='Use original rewards if available.')
    parser.add_argument('--window_length', type=int, default=5, help="Window length for video frame embeddings.")

    args = parser.parse_args()

    print("Loading trajectories...")
    with h5py.File(args.trajs_to_label, 'r') as traj_file, h5py.File(args.output, 'w') as output_file:
        print("Processing and saving trajectories iteratively...")
        label_trajectories_iteratively(args, traj_file, output_file)


        for key in traj_file['0'].keys():
            if key not in ['rewards', 'img']:
                print(f"Saving {key}...")
                items = []
                for i in range(len(traj_file.keys())):
                    items.extend(traj_file[str(i)][key])

                try:
                    array_data = np.array(items)
                    output_file.create_dataset(key, data=array_data, dtype=array_data.dtype)
                except:
                    print(f"Could not save {key}...")
                    breakpoint()

    print(f"Trajectories with rewards saved to {args.output}")


if __name__ == "__main__":
    main()
