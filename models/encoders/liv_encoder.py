from models.encoders.base_encoder import BaseEncoder
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F
from models.reward_model.self_attention_utils import MultiHeadAttentionSubtraction, MultiHeadAttention
from liv import load_liv
import clip



class LIVEncoder(BaseEncoder):
    def __init__(self, model_load_path: str, use_pca: bool, attention_heads: int, device: str = 'cuda', batch_size=64):
        """
        Initializes the LIV reward model.
        :param model_load_path: Path to the model checkpoint.
        :param use_pca: Whether to use PCA for the video embeddings.
        :param attention_heads: Number of attention heads to use in the transformer model.
        :param pca_model_dir: Path to the PCA model checkpoint directory where the `pca_text.pkl` and `pca_video.pkl` files are located.
        :param device: Device to run the model on (default: 'cuda').
        :param batch_size: Batch size to use for encoding data (default: 64).
        """
        super().__init__(device, batch_size)
        self.use_pca = use_pca
        self.attention_heads = attention_heads
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
        image_embeddings = image_embeddings.cpu().numpy()
        return image_embeddings

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
        return 'LIVEncoder'