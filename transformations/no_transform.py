import torch
import abc

class NoTransform(abc.ABC):
    def __init__(self, args):
        """
        Initialize the encoder. Subclasses can implement specific initialization as needed.
        """
        pass

    def apply_transform(self, embedding):
        """
        Encodes a text input into a representation.
        :param text: Text data to be encoded.
        :return: Encoded representation of the text.
        """
        return embedding
