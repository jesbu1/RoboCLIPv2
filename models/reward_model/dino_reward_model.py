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
from models.encoders.dino_miniLM_encoder import Dino_miniLM_Encoder

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

class DINORewardModel(BaseRewardModel):
    def __init__(self, use_pca: bool, device: str = 'cuda', dino_batch_size=32, max_num_frames_per_episode=32, batch_size=128, success_bonus: float = 10.0):
        """
        Initializes the LIV reward model.
        :param use_pca: Whether to use PCA for the video embeddings.
        :param device: Device to run the model on (default: 'cuda').
        :param dino_batch_size: Batch size to use for encoding data (default: 32).
        :param max_num_frames_per_episode: Maximum number of frames per episode (default: 32).
        :param batch_size: Batch size to use for encoding data (default: 128).
        """
        super().__init__(device, batch_size, success_bonus=success_bonus)
        # self.use_pca = use_pca
        # self.attention_heads = attention_heads
        # self.pretrained_liv_model = self._load_model(model_load_path)

        self.dino_encoder = Dino_miniLM_Encoder(use_pca, device, dino_batch_size=dino_batch_size, max_num_frames_per_episode=max_num_frames_per_episode, batch_size=batch_size)

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """

        return self.dino_encoder.encode_text(text)

    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """

        return self.dino_encoder.encode_images(images)

    def _calculate_reward_batch(self, encoded_texts: np.ndarray, encoded_videos: np.ndarray) -> np.ndarray:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        pass

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        return self.dino_encoder.img_output_dim
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return self.dino_encoder.text_output_dim
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'DINORewardModel'