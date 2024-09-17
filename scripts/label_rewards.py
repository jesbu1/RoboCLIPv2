import argparse
import torch
import torch as th
import h5py
from tqdm import tqdm

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from encoders import XCLIPEncoder, S3DEncoder, VLCEncoder  # Assume these are imported or defined elsewhere
# from transformation_model import load_transformation_model  # Assume this loads the transformation model


class SingleLayerMLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


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
            self.transform_model = SingleLayerMLP(512, 512, normalize=True)
            dict = torch.load(args.transform_model_path)
            self.transform_model.load_state_dict(dict['model_state_dict'])
            self.transform_model.eval().cuda()

    def label_trajectories(self, traj_data):
        """
        Densely labels rewards for each trajectory based on encoded video frames.
        :param traj_data: Trajectory data containing video frames to be labeled.
        :return: Rewards for each trajectory.
        """
        all_rewards = []

        # Iterate over each trajectory in the dataset
        for traj in tqdm(traj_data):
            buffer_of_images = np.array(traj['images'])
            rewards = []

            text_embedding = self.encoder.encode_text(traj['instruction'][0])

            # Process image buffer as growing video
            for t in range(1, len(buffer_of_images) + 1):
                video_frames = buffer_of_images[:t]
                video_embedding = self.encoder.encode_video(video_frames)

                # Apply the transformation model if available
                if hasattr(self, 'transform_model'):
                    video_embedding = self.transform_model(video_embedding)

                # Calculate reward (similarity between video embedding and text embedding)
                similarity = self.compute_similarity(video_embedding, text_embedding)
                rewards.append(similarity)

            # Store rewards for the current trajectory
            all_rewards.append(rewards)

            # FOR TESTING: Stop after processing two trajectories
            if len(all_rewards) == 2:
                break

        return all_rewards

    def compute_similarity(self, video_embedding, text_embedding):
        """
        Computes similarity between video embedding and a precomputed text embedding.
        :param video_embedding: Encoded video frames.
        :return: Similarity score as reward.
        """
        similarity = torch.matmul(text_embedding, video_embedding.t()).cpu().numpy()[0][0]
        return similarity


def parse_trajectories(h5_file, chunk_size=1000):
    actions = h5_file['action']
    dones = h5_file['done']
    images = h5_file['img']
    states = h5_file['state']
    strings = h5_file['string']
    
    trajectories = []
    current_trajectory = {'actions': [], 'states': [], 'images': [], 'instruction': []}
    
    # Initialize chunk indices
    start_idx = 0
    end_idx = min(chunk_size, len(dones))

    while start_idx < len(dones):
        # Load a chunk of data
        chunk_actions = actions[start_idx:end_idx]
        chunk_dones = dones[start_idx:end_idx]
        chunk_images = images[start_idx:end_idx]
        chunk_states = states[start_idx:end_idx]
        chunk_strings = strings[start_idx:end_idx]
        
        for i in range(len(chunk_dones)):
            if chunk_dones[i]:
                # End of a trajectory
                current_trajectory['actions'].append(chunk_actions[i])
                current_trajectory['states'].append(chunk_states[i])
                current_trajectory['images'].append(chunk_images[i])
                current_trajectory['instruction'].append(chunk_strings[i])
                
                trajectories.append(current_trajectory)
                current_trajectory = {'actions': [], 'states': [], 'images': [], 'instruction': []}
            else:
                # Continuation of the current trajectory
                current_trajectory['actions'].append(chunk_actions[i])
                current_trajectory['states'].append(chunk_states[i])
                current_trajectory['images'].append(chunk_images[i])
                current_trajectory['instruction'].append(chunk_strings[i])
        
        # Move to the next chunk
        start_idx = end_idx
        end_idx = min(start_idx + chunk_size, len(dones))

    # Handle the last trajectory if it doesn't end with a "done" flag
    if len(current_trajectory['actions']) > 0:
        trajectories.append(current_trajectory)
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Label rewards for trajectories.")
    # data
    parser.add_argument('--trajs_to_label', required=True, help='Path to the trajectories file (HDF5 format).')
    parser.add_argument('--output', required=True, help='Path to save the updated trajectories.')

    # encoder
    parser.add_argument('--encoder_type', required=False, choices=['xclip', 's3d', 'vlc'], help='Type of encoder to use.', default='xclip')
    parser.add_argument('--encoder_path', required=False, help='Path to the encoder model file.')

    # transformation model
    parser.add_argument('--transform_model_path', required=False, help='Path to the transformation model file.', default=None)

    args = parser.parse_args()

    # Initialize RewardLabeler with arguments
    reward_labeler = RewardLabeler(args)

    print("Loading trajectories...")
    with h5py.File(args.trajs_to_label, 'r') as traj_file:
        traj_data = parse_trajectories(traj_file)

    print(f"Labeling rewards for trajectories in with size {len(traj_data)}...")
    # Label rewards for the trajectories
    all_rewards = reward_labeler.label_trajectories(traj_data)

    # Save the updated trajectories with rewards to a new HDF5 file
    with h5py.File(args.output, 'w') as output_file:
        for i, traj in enumerate(traj_data):
            traj_id = f'trajectory_{i}'
            grp = output_file.create_group(traj_id)
            grp.create_dataset('actions', data=np.array(traj['actions']))
            grp.create_dataset('states', data=np.array(traj['states']))
            grp.create_dataset('images', data=np.array(traj['images']))
            grp.create_dataset('instruction', data=np.array(traj['instruction']))
            grp.create_dataset('rewards', data=np.array(all_rewards[i]))

            # in case all_rewards is not the same length as traj_data
            if i == len(all_rewards) - 1:
                break

    print(f"Trajectories with rewards saved to {args.output}")

if __name__ == "__main__":
    main()