# from test_scripts.test_iql import parse_reward_model
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.reward_model.liv_reward_model import LIVRewardModel
import torch as th
import numpy as np


def main():

    reward_model = LIVRewardModel(
            model_load_path='snapshot_10000.pt',
            use_pca=False,
            attention_heads=4,
            pca_model_dir=None,
            batch_size=64,
            success_bonus=200,
        )
    text_instruction = "Reach the goal"
    frames = th.randn(1, 128, 224, 224, 3)
    lang_feat_reward = reward_model.encode_text(text_instruction).squeeze()
    reward_language_features = (
                th.Tensor(lang_feat_reward)
                .float()
                .to(reward_model.device)
                .unsqueeze(0)
    )
    # print(f"frames shape: {frames.shape}") # (1, 128, 224, 224, 3)
    frames_embeddings = th.from_numpy(reward_model.encode_images(
        frames
    )).unsqueeze(0)
    import pdb ; pdb.set_trace()
    # print(f"frames_embeddings shape: {frames_embeddings.shape}") # (1, 32, 768)
    reward = reward_model.calculate_rewards(
        reward_language_features, frames_embeddings
    )
    print(f"reward: {reward}")
    # exit()

if __name__ == "__main__":
    main()