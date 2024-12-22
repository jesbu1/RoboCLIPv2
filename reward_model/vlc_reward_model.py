import torch
import numpy as np

import abc
from typing import Union
from reward_model.base_reward_model import BaseRewardModel
# TODO: fill in VLCEncoder


class VLCRewardModel(BaseRewardModel):
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """
        Initializes the VLC encoder with a pretrained model.
        :param model_path: Path to the VLC model file.
        :param device: Device to run the model on.
        """
        super().__init__(device)
        self.model = self.load_model(model_path)
        self.model.eval()  # Set model to evaluation mode

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Loads the pretrained VLC model from the provided path.
        :param model_path: Path to the pretrained model file.
        :return: Loaded model.
        """
        model = torch.load(model_path)
        return model

    def encode_text(self, text: Union[str, list]) -> np.ndarray:
        """
        Encodes text input using VLC.
        :param text: Input text.
        :return: Encoded text representation.
        """
        # Example: Assume self.model has a method for text encoding
        with torch.no_grad():
            encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_video(self, video_frames: np.ndarray) -> np.ndarray:
        """
        Encodes video frames using VLC.
        :param video_frames: Sequence of video frames.
        :return: Encoded video representation.
        """
        # Example: Assume self.model has a method for video encoding
        with torch.no_grad():
            encoded_video = self.model.encode_video(video_frames)
        return encoded_video
