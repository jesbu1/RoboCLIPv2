import os, sys

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from encoders.encoder import BaseEncoder




class XCLIPEncoder(BaseEncoder):
    def __init__(self):
        """
        Initializes the X-CLIP encoder using a pretrained model.
        """
        super().__init__()
        self.tokenizer, self.model, self.processor = self.load_model()

    def load_model(self):
        """
        Loads the pretrained X-CLIP model and its components.
        :return: X-CLIP tokenizer, model, and processor.
        """
        model_name = "microsoft/xclip-base-patch16-zero-shot"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).cuda()
        processor = AutoProcessor.from_pretrained(model_name)
        return tokenizer, model, processor

    def normalize_embeddings(self, embeddings, return_tensor=True):
        """
        Normalizes the embeddings using L2 normalization.
        :param embeddings: The embeddings to be normalized.
        :param return_tensor: Whether to return the result as a tensor or numpy array.
        :return: Normalized embeddings.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        if return_tensor:
            return normalized_embeddings
        else:
            return normalized_embeddings.detach().cpu().numpy()

    def encode_text(self, text):
        """
        Encodes text input using X-CLIP.
        :param text: Input text (list of strings).
        :return: Normalized text embeddings.
        """

        # if string is a byte or numpy byte, convert to string
        if isinstance(text, bytes) or isinstance(text, np.bytes_):
            text = text.decode()


        text_tokens = self.tokenizer(text, return_tensors="pt")
        for key in text_tokens:
            text_tokens[key] = text_tokens[key].cuda()

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_tokens)

        return self.normalize_embeddings(text_features)

    def encode_video(self, video_frames):
        """
        Encodes video input using X-CLIP.
        :param video_frames: Input video frames (as a numpy array of shape [frames, height, width, channels]).
        :return: Normalized video embeddings.
        """

        video_frames = self.adjust_frames_xclip(video_frames)

        video_input = self.processor(videos=list(video_frames), return_tensors="pt")
        video_input = video_input["pixel_values"].cuda()

        with torch.no_grad():
            video_features = self.model.get_video_features(video_input)

        return self.normalize_embeddings(video_features)


    def adjust_frames_xclip(self, frames, target_frame_count=32):
        """
        Ensures same numbers of frames(32). returns a numpy array of shape (target_frame_count, 224, 224, 3)
        """
        frames = np.array(frames)
        frame_count = frames.shape[0]
        #print(f"frames number{frame_count}")
        # frames = th.from_numpy(frames)

        if len(frames) > target_frame_count:
            index = np.linspace(0, len(frames)-1, target_frame_count, dtype=int)
            frames = frames[index]
        elif len(frames) < target_frame_count:
            last_frame = frames[-1]
            last_frame = np.expand_dims(last_frame, axis=0)
            for _ in range(target_frame_count - len(frames)):
                frames = np.concatenate([frames, last_frame])
            
        # TODO: Not sure what this is meant to do!
        # frames = frames[:,240-112:240+112,320-112:320+112,:]
        # # frames = frames[None, :,:,:,:]
        # frames = processor(videos=list(frames), return_tensors="pt")
        # frames = frames["pixel_values"]
        return frames


if __name__ == "__main__":
    # Example usage of the XCLIPEncoder class
    encoder = XCLIPEncoder()

    # Encoding text
    text_samples = ["a", "b", "c"]
    text_embeddings = encoder.encode_text(text_samples)
    print(f"Text embeddings shape: {text_embeddings.shape}")  # (3, 512)

    # Encoding video
    video_sample = np.random.randint(0, 255, (32, 224, 224, 3))  # Example random video frames
    video_embeddings = encoder.encode_video(video_sample)
    print(f"Video embeddings shape: {video_embeddings.shape}")  # (1, 512)