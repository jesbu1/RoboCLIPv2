from reward_model import BaseRewardModel
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F
from reward_model.self_attention_utils import MultiHeadAttentionSubtraction, MultiHeadAttention
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
    def __init__(self, model_load_path: str, device: str = 'cuda', batch_size=64, success_bonus: float = 10.0, reward_at_every_step: bool = False, use_pca: bool = False):
        """
        Initializes the LIV reward model.
        :param model_load_path: Path to the model checkpoint.
        :param device: Device to run the model on (default: 'cuda').
        :param batch_size: Batch size to use for encoding data (default: 64).
        :param reward_at_every_step: Whether to calculate rewards at every step (default: False).
        """
        super().__init__(device, batch_size, success_bonus=success_bonus)
        self.use_pca = use_pca
        self.reward_at_every_step = reward_at_every_step
        self.pretrained_liv_model = self._load_model(model_load_path)


    def _load_model(self, model_load_path: str) -> torch.nn.Module:
        """
        Loads the pretrained LIV model from the provided path.
        :param model_load_path: Path to the pretrained model file.
        :return: Loaded model.
        """
        #TODO: add support for loading the finetuned model
        model = load_liv()
        state_dict = torch.load(model_load_path)["liv"]
        model.module.load_state_dict(state_dict)
        return model.to(self.device)

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        text = clip.tokenize(text)
        with torch.no_grad():
            text_embeddings = self.pretrained_liv_model(input=text, modality="text")
        # text_embeddings = normalize_embeddings(text_embeddings, return_tensor=True)
        return text_embeddings.detach().cpu().numpy()

    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        assert images.shape[0] == 1, "LIV doesn't support batch > 1"
        images = images.squeeze(0)
        with torch.no_grad():
            image_embeddings = self.pretrained_liv_model(input=images, modality="vision")
        # image_embeddings = normalize_embeddings(image_embeddings, return_tensor=True)
        return image_embeddings.unsqueeze(0)

    def _calculate_reward_batch(self, encoded_texts: np.ndarray, encoded_videos: np.ndarray) -> np.ndarray:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        # Accept both numpy and torch tensors
        if isinstance(encoded_texts, np.ndarray):
            encoded_texts = torch.tensor(encoded_texts, dtype=torch.float32, device=self.device)
        if isinstance(encoded_videos, np.ndarray):
            encoded_videos = torch.tensor(encoded_videos, dtype=torch.float32, device=self.device)

        # Shapes:
        #   encoded_texts: (batch, 1, text_dim) or (batch, text_dim)
        #   encoded_videos: (batch, T, img_dim)
        if encoded_texts.dim() == 3:
            encoded_texts = encoded_texts.squeeze(1)

        assert encoded_videos.dim() == 3, "encoded_videos should be (batch, T, dim)"
        # For LIV, we compute cosine similarity between text and the last video frame embedding
        # encoded_videos shape: (batch=1, T, dim)
        # encoded_texts shape: (batch=1, dim)
        #print(encoded_videos.shape)
        # print(encoded_texts.shape)
        # Take the last frame embedding as the final state
        final_video_emb = encoded_videos[:, -1, :]  # (batch=1, dim)
        
        # Calculate cosine similarity
        similarities = F.cosine_similarity(final_video_emb, encoded_texts, dim=1)  # (batch=1,)
        #print(similarities)

        # Return as numpy array to match base class interface
        return similarities.detach().cpu().numpy()

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        if self.use_pca:
            return self.pca_video_model.components_.shape[0]
        return 1024 # for LIV
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        if self.use_pca:
            return self.pca_text_model.components_.shape[0]
        return 1024 # for LIV
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'LIVRewardModel'