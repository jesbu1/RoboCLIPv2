import torch

import abc
from encoders.encoder import BaseEncoder



class S3DEncoder(BaseEncoder):
    def __init__(self, model_path):
        """
        Initializes the S3D encoder with a pretrained model.
        :param model_path: Path to the S3D model file.
        """
        super().__init__()
        self.model = self.load_model(model_path)
        self.model.eval()  # Set model to evaluation mode

    def load_model(self, model_path):
        """
        Loads the pretrained S3D model from the provided path.
        :param model_path: Path to the pretrained model file.
        :return: Loaded model.
        """
        model = torch.load(model_path)
        return model

    def encode_text(self, text):
        """
        Encodes text input using S3D.
        :param text: Input text.
        :return: Encoded text representation.
        """
        # Example: Assume self.model has a method for text encoding
        with torch.no_grad():
            encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_video(self, video_frames):
        """
        Encodes video frames using S3D.
        :param video_frames: Sequence of video frames.
        :return: Encoded video representation.
        """
        # Example: Assume self.model has a method for video encoding
        with torch.no_grad():
            encoded_video = self.model.encode_video(video_frames)
        return encoded_video
