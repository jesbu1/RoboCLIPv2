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

    def label_trajectories(self, traj_data):
        """
        Densely labels rewards for each trajectory based on encoded video frames.
        :param traj_data: Trajectory data containing video frames to be labeled.
        :return: Rewards for each trajectory.
        """
        all_rewards = []
        all_lang_embeds = []

        image = traj_data['img']
        previous_instruction = None
        instruction = traj_data['string']
        state = traj_data['state']
        done = traj_data['done']

        images = []
        timesteps = []
        current_timestep = 0
        for i in tqdm(range(len(done))):

            

            if instruction[i] != previous_instruction:
                text_embedding = self.encoder.encode_text(instruction[i])
                previous_instruction = instruction[i]

            # We append images to the buffer, until we reach the end of the trajectory
            images.append(image[i])
            all_lang_embeds.append(text_embedding.cpu().numpy())
            timesteps.append(current_timestep)

            if done[i]:

                video_frames = np.array(images)
                video_embedding = self.encoder.encode_video(video_frames)

                # Apply the transformation model if available
                if hasattr(self, 'transform_model'):
                    video_embedding = self.transform_model.apply_transform(video_embedding)

                # Calculate reward (similarity between video embedding and text embedding)
                similarity = self.compute_similarity(video_embedding, text_embedding)
                all_rewards.append(similarity)

                # Then we reset the buffer
                images = []

                current_timestep = 0

            else:
                all_rewards.append(0.0)

                current_timestep += 1

            if i == 100:
                break

        return all_rewards, all_lang_embeds, timesteps

    def compute_similarity(self, video_embedding, text_embedding):
        """
        Computes similarity between video embedding and a precomputed text embedding.
        :param video_embedding: Encoded video frames.
        :return: Similarity score as reward.
        """
        similarity = torch.matmul(text_embedding, video_embedding.t()).cpu().numpy()[0][0]
        return similarity

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
        traj_data = traj_file

        print(f"Labeling rewards for trajectories in with size {len(traj_data)}...")
        # Label rewards for the trajectories
        all_rewards, all_lang_embeds, timesteps = reward_labeler.label_trajectories(traj_data)


        print(f"Saving trajectories with rewards to {args.output}...")
        # Save an updated version of the trajectories with rewards
        with h5py.File(args.output, 'w') as output_file:
            # Save each of the keys along with the corresponding data
            output_file.create_dataset('rewards', data=all_rewards, dtype='float32')
            output_file.create_dataset('lang_embedding', data=all_lang_embeds, dtype='float32')
            output_file.create_dataset('timesteps', data=timesteps, dtype='int32')
            # copy the rest of the data from traj_data to output_file
            for key in traj_data.keys():
                if key != 'rewards':
                    output_file.create_dataset(key, data=traj_data[key], dtype=traj_data[key].dtype)



    print(f"Trajectories with rewards saved to {args.output}")

if __name__ == "__main__":
    main()
