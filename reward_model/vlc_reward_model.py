import torch
import numpy as np

import abc
from typing import Union
from models.reward_model.base_reward_model import BaseRewardModel
import requests
import pickle
import base64
import io
import os
import imageio
import cv2
from typing import List, Union
# TODO: fill in VLCEncoder


class VLCRewardModel(BaseRewardModel):
    def __init__(self, server_url: str = "http://10.137.28.55:5000", batch_size: int = 64, success_bouns: int = 10, device: str = "cuda", success_bonus: int = 10, reward_at_every_step: bool = False) -> None:
        """
        Initializes the VLC reward client that communicates with a remote VLC server.
        :param server_url: URL of the VLC reward calculation server (Flask).
        :param device: Device to run any necessary local computations on.
        """
        super().__init__(device= device, batch_size= batch_size, success_bonus= success_bonus)
        self.server_url = server_url  # server address
        self.reward_at_every_step = reward_at_every_step
        self.success_bonus = success_bonus


    def send_to_server(self, video_frames: np.ndarray, text: str) -> float:
        """
        Sends video frames and text to the VLC server for reward calculation.
        :param video_frames: Numpy array containing all video frames.
        :param text: Corresponding textual description.
        :return: Computed reward.
        """
        print(video_frames.shape)
        print(text)
        exit()
        video_bytes = io.BytesIO()
        np.save(video_bytes, video_frames)
        video_encoded = base64.b64encode(video_bytes.getvalue()).decode('utf-8')

        request_data = {
            "video": video_encoded,
            "text": text
        }
        request_data = pickle.dumps(request_data)

        response = requests.post(f"{self.server_url}/compute_reward", data=request_data)
        response.raise_for_status()

        reward = response.json()["reward"]
        return reward

    def calculate_rewards(self, video_frames: np.ndarray, text: str) -> float:
        """
        Calls the VLC server to compute the video-language reward.
        :param video_frames: Sequence of video frames.
        :param text: Corresponding textual description.
        :return: Computed reward.
        """
        return self.send_to_server(video_frames, text)
    
    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        with torch.no_grad():
            encoded_input = self.minilm_tokenizer(
                text, padding=False, truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.minilm_model(**encoded_input)
            text_embeddings = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .numpy()
            )

        return text_embeddings

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_images, height, width, channels).
        :return: Encoded representation of each frame.
        """
        # TODO: this can probably handle multiple batches but untested
        assert images.shape[0] == 1, "LIV doesn't support batch > 1"
        images = images.squeeze(0)

        with torch.inference_mode():
            episode_images_dino = [
                dino_load_image(
                    (img.to("cpu").numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                for img in images
            ]
            episode_images_dino = [
                torch.concatenate(episode_images_dino[i : i + self.dino_batch_size])
                for i in range(0, len(episode_images_dino), self.dino_batch_size)
            ]
            embedding_list = []
            for batch in episode_images_dino:
                episode_image_embeddings = (
                    self.dino_vits14(batch.to(self.device)).squeeze().detach().cpu()
                )
                embedding_list.append(episode_image_embeddings)
            episode_image_embeddings = torch.concat(embedding_list)

        return episode_image_embeddings.unsqueeze(0)
    
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'VLCRewardModel'

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        return 768 # for LIV
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return 384 # for LIV

def read_video_as_frames(video_path: str, as_tensor: bool = False) -> Union[List[np.ndarray], torch.Tensor]:

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = []
    ext = os.path.splitext(video_path)[1].lower()

    if ext == ".gif":
        reader = imageio.get_reader(video_path)
        for frame in reader:
            frames.append(frame)
        reader.close()
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()

    if as_tensor:
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
        return frames_tensor

    return frames


if __name__ == "__main__":
    reward_model = VLCRewardModel()
    video_frames = np.random.rand(100, 224, 224, 3)
    video_frames = read_video_as_frames("/scr/yusenluo/RoboCLIP/self_collected_vids/window_close/GT/1.mp4", as_tensor=False)
    text = "closing the window"
    reward = reward_model.calculate_rewards(video_frames, text)
    print(reward)