import torch
import abc
import numpy as np
from typing import List, Union

class BaseEncoder(abc.ABC):
    def __init__(self, device: str = 'cuda', batch_size=64):
        """
        Initialize the encoder. Subclasses can implement specific initialization as needed.
        """
        self.device = torch.device(device)
        self.batch_size = batch_size

    def encode_text(self, text: Union[str, List]) -> np.ndarray:
        """
        Encodes a text input into a representation.
        :param text: Text data to be encoded. If a list of strings is provided, it will be batch encoded.
        :return: Encoded representation of the text.
        """
        if isinstance(text, list):
            for i in range(0, len(text), self.batch_size):
                batch_text = text[i:i+self.batch_size]
                encoded_text = self._encode_text_batch(batch_text)
                if i == 0:
                    encoded_text_all = encoded_text
                else:
                    encoded_text_all = np.concatenate((encoded_text_all, encoded_text))
        else:
            encoded_text_all = self._encode_text_batch([text])
        return encoded_text_all
    
    @abc.abstractmethod
    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        pass

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encodes a video input (sequence of frames) into an image representation.
        :param images: A sequence of video frames to be encoded. The shape of the input should be (num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            batch_images = torch.tensor(batch_images, dtype=torch.float32).to(self.device)
            encoded_images = self._encode_image_batch(batch_images)
            if i == 0:
                encoded_images_all = encoded_images
            else:
                encoded_images_all = np.concatenate((encoded_images_all, encoded_images))
        return encoded_images_all
    
    @abc.abstractmethod
    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        pass

    def calculate_rewards(self, encoded_texts: np.ndarray, encoded_videos: np.ndarray) -> np.ndarray:
        """
        Calculates the rewards for given text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        for i in range(0, len(encoded_texts), self.batch_size):
            batch_texts = encoded_texts[i:i+self.batch_size]
            batch_videos = encoded_videos[i:i+self.batch_size]
            rewards = self._calculate_reward_batch(batch_texts, batch_videos)
            if i == 0:
                rewards_all = rewards
            else:
                rewards_all = np.concatenate((rewards_all, rewards))
        return rewards_all
    
    @abc.abstractmethod
    def _calculate_reward_batch(self, encoded_texts: np.ndarray, encoded_videos: np.ndarray) -> np.ndarray:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        pass

    @property
    def output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        pass