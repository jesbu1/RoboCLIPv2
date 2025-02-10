import torch
import numpy as np

import abc
from encoders.encoder import BaseEncoder
from typing import List, Union
# TODO: fill this in


class S3DEncoder(BaseEncoder):
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """
        Initializes the S3D encoder with a pretrained model.
        :param model_path: Path to the S3D model file.
        :param device: Device to run the model on.
        """
        super().__init__(device)
        self.model = self.load_model(model_path)
        self.model.eval()  # Set model to evaluation mode

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Loads the pretrained S3D model from the provided path.
        :param model_path: Path to the pretrained model file.
        :return: Loaded model.
        """
        model = torch.load(model_path)
        return model

    def encode_text(self, text: Union[str, np.ndarray]) -> np.ndarray:
        """
        Encodes text input using S3D.
        :param text: Input text.
        :return: Encoded text representation.
        """
        # Example: Assume self.model has a method for text encoding
        with torch.no_grad():
            encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encodes video frames using S3D.
        :param video_frames: Sequence of video frames.
        :return: Encoded video representation.
        """
        # Example: Assume self.model has a method for video encoding
        with torch.no_grad():
            encoded_video = self.model.encode_video(images)
        return encoded_video
