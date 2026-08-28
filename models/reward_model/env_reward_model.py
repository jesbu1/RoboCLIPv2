import torch
import numpy as np

import abc
from typing import Union, List
from models.reward_model.base_reward_model import BaseRewardModel
from models.encoders.dino_miniLM_encoder import Dino_miniLM_Encoder

class EnvRewardModel(BaseRewardModel):
    def __init__(self, reward_type: str="dense", model_path: str = "", device: str = "cuda", reward_at_every_step: bool = False, success_bonus: float = 10.) -> None:
        """
        Env reward model simply passes the reward from the simulator.
        Initializes a LIV encoder with a pretrained model 
        for image and text encoding.
        :param model_path: Path to the LIV model file.
        :param device: Device to run the model on.
        """
        super().__init__(device, success_bonus=success_bonus)

        self.reward_type = reward_type

        # TODO: Turn this into a cfg option and a param in every constructor
        self.reward_at_every_step = reward_at_every_step

        self.dino_encoder = Dino_miniLM_Encoder(use_pca=False, device=device, dino_batch_size=128, max_num_frames_per_episode=128, batch_size=64)

    
    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        return self.dino_encoder._encode_text_batch(text)
    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        return self.dino_encoder._encode_image_batch(images)

    def _calculate_reward_batch(self, encoded_texts, encoded_videos):
        """
        Calculates the reward for a batch of encoded texts and videos.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward for the batch.
        """
        return 0 # Always return 0 reward
    
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return self.reward_type
    
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
