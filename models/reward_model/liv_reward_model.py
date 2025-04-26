from models.reward_model import BaseRewardModel
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F
from models.reward_model.self_attention_utils import MultiHeadAttentionSubtraction, MultiHeadAttention
from models.encoders.liv_encoder import LIVEncoder
from liv import load_liv
import clip


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

class LIVRewardModel(BaseRewardModel):
    def __init__(self, model_load_path: str, use_pca: bool, attention_heads: int, pca_model_dir: str = None, device: str = 'cuda', batch_size=64, reward_at_every_step: bool = False, success_bonus: int = 10, last_frame_reward_only: bool = True):
        """
        Initializes the LIV reward model.
        :param model_load_path: Path to the model checkpoint.
        :param use_pca: Whether to use PCA for the video embeddings.
        :param attention_heads: Number of attention heads to use in the transformer model.
        :param pca_model_dir: Path to the PCA model checkpoint directory where the `pca_text.pkl` and `pca_video.pkl` files are located.
        :param device: Device to run the model on (default: 'cuda').
        :param batch_size: Batch size to use for encoding data (default: 64).
        """
        super().__init__(device, batch_size, success_bonus=success_bonus)
        # self.use_pca = use_pca
        # self.attention_heads = attention_heads
        # self.pretrained_liv_model = self._load_model(model_load_path)
        self.reward_at_every_step = reward_at_every_step
        self.liv_encoder = LIVEncoder(model_load_path, use_pca, attention_heads, device, batch_size)
        self.last_frame_reward_only = last_frame_reward_only
        self.reward_at_every_step = reward_at_every_step

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        return self.liv_encoder.encode_text(text)

    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        # print(f"LIV images shape: {images.shape}")
        return self.liv_encoder.encode_images(images)

    def _calculate_reward_batch(self, encoded_texts: np.ndarray, encoded_videos: np.ndarray) -> np.ndarray:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        # print(f"encoded_texts shape: {encoded_texts.shape}")
        # print(f"encoded_videos shape: {encoded_videos.shape}")
        similarities = F.cosine_similarity(encoded_videos.squeeze(0), encoded_texts, dim=1)
        if self.last_frame_reward_only:
            final_reward = similarities[-1]
            final_reward = float(final_reward.detach().cpu().item())
        else:
            final_reward = similarities.detach().cpu().numpy()
        
        return final_reward
    
    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        return self.liv_encoder.img_output_dim
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return self.liv_encoder.text_output_dim
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'LIVRewardModel'