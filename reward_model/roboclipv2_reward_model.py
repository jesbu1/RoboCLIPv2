from reward_model import BaseRewardModel
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F
from reward_model.self_attention_utils import MultiHeadAttentionSubtraction, MultiHeadAttention
from reward_model.liv_reward_model import LIVRewardModel
from liv import load_liv
import clip

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

class RoboclipV2RewardModel(BaseRewardModel):
    def __init__(self, model_load_path: str, use_pca: bool, attention_heads: int, pca_model_dir: str = None, device: str = 'cuda', batch_size=64, reward_at_every_step: bool = False):
        """
        Initializes the RoboclipV2 reward model.
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
        self.pretrained_liv_model, self.pca_video_model, self.pca_text_model, self.video_encoding_model = self._load_model(model_load_path, pca_model_dir)
        self.reward_at_every_step = reward_at_every_step

    def _load_model(self, model_load_path: str, pca_model_path: str = None):
        #TODO: add support for loading the finetuned model
        liv_model = load_liv()
        liv_model.eval()
        pca_video_model = None
        pca_text_model = None
        if pca_model_path is not None:
            pca_text_path = os.path.join(pca_model_path, 'pca_text.pkl') 
            pca_video_path = os.path.join(pca_model_path, 'pca_video.pkl') 
            pca_text_model = joblib.load(pca_text_path)
            pca_video_model = joblib.load(pca_video_path)
            pca_dim = pca_video_model.components_.shape[0]
            video_encoding_model = MultiHeadAttentionSubtraction(pca_dim, num_heads=self.attention_heads)
        else:
            # 1024 is hardcoded for LIV's output dimension
            video_encoding_model = MultiHeadAttentionSubtraction(1024, num_heads=self.attention_heads)
        dict = torch.load(model_load_path)
        if 'model_state_dict' in dict.keys():
            video_encoding_model.load_state_dict(dict["model_state_dict"])
        else:
            video_encoding_model.load_state_dict(dict)
        video_encoding_model = video_encoding_model.eval()
        return liv_model.to(self.device), pca_video_model, pca_text_model, video_encoding_model.to(self.device)

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_embeddings = self.pretrained_liv_model(input=text, modality="text")
            if self.use_pca:
                text_embeddings = self.pca_text_model.transform(text_embeddings)
        text_embeddings = normalize_embeddings(text_embeddings)
        return text_embeddings.detach().cpu().numpy()

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_images, height, width, channels).
        :return: Encoded representation of each frame.
        """
        assert images.shape[0] == 1, "LIV doesn't support batch > 1"
        images = images.squeeze(0)
        with torch.no_grad():
            image_embeddings = self.pretrained_liv_model(input=images, modality="vision")
            if self.pca_video_model:
                image_embeddings = self.pca_video_model.transform(image_embeddings.cpu().numpy())
                image_embeddings = torch.from_numpy(image_embeddings).float().to(self.device)
        image_embeddings = normalize_embeddings(image_embeddings, return_tensor=True)
        return image_embeddings.unsqueeze(0)
        

    def _calculate_reward_batch(self, encoded_texts: torch.Tensor, encoded_videos: torch.Tensor) -> torch.Tensor:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations. Shape: (batch_size, num_images, embedding_dim).
        :return: Reward values for each text-video pair.
        """
        encoded_texts = encoded_texts.squeeze(0) # remove batch dimension for video_encoding_model not supported and then only 
        # TODO: add the processing for downsampling if needed @Yusen @Jiahui
        reward = self.video_encoding_model(encoded_videos.float(), None, encoded_texts.float()).item()
        return reward

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
        return 'RoboclipV2RewardModel'