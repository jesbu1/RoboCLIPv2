import torch
import abc

class BaseEncoder(abc.ABC):
    def __init__(self):
        """
        Initialize the encoder. Subclasses can implement specific initialization as needed.
        """
        pass

    @abc.abstractmethod
    def encode_text(self, text):
        """
        Encodes a text input into a representation.
        :param text: Text data to be encoded.
        :return: Encoded representation of the text.
        """
        pass

    @abc.abstractmethod
    def encode_video(self, video_frames):
        """
        Encodes a video input (sequence of frames) into a representation.
        :param video_frames: A sequence of video frames to be encoded.
        :return: Encoded representation of the video.
        """
        pass
