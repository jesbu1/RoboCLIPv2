import torch
import numpy as np

import abc
from typing import Union
from reward_model import BaseRewardModel
import requests
import pickle
import base64
import io
import os
import imageio
import cv2
from typing import List, Union
# TODO: fill in VLCEncoder

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from reward_model.clip_utils import dino_load_image, mean_pooling
import torchvision


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class VLCRewardModel(BaseRewardModel):
    def __init__(
        self,
        server_url: str = "http://10.137.28.55:5001",
        batch_size: int = 64,
        success_bouns: int = 10,
        device: str = "cuda",
        success_bonus: int = 10,
        reward_at_every_step: bool = False,
    ) -> None:
        """
        Initializes the VLC reward client that communicates with a remote VLC server.
        :param server_url: URL of the VLC reward calculation server (Flask).
        :param device: Device to run any necessary local computations on.
        """
        super().__init__(
            device=device, batch_size=batch_size, success_bonus=success_bonus
        )
        self.server_url = server_url  # server address
        self.reward_at_every_step = reward_at_every_step
        self.success_bonus = success_bonus

        # Load minilm-12v2
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )

        # for the image embedding, we use dino
        self.dino_vits14 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14"
        ).to(device)

        self.dino_batch_size = 64

    def encode_text(self, text: Union[str, list]) -> np.ndarray:
        """
        Since VLC just calls a server, we just return the same thing again
        """
        return text

    def encode_text_for_policy(self, text: Union[str, List]) -> np.ndarray:
        """
        Encodes a text input into a representation for policy training.
        :param text: Text data to be encoded. If a list of strings is provided, it will be batch encoded.
        :return: Encoded representation of the text.
        """

        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            text_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

        # normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        return text_embeddings.detach().cpu().numpy()

    def encode_images_for_policy(self, images: np.ndarray) -> np.ndarray:
        """
        Encodes a video input (sequence of frames) into an image representation.
        :param images: A sequence of video frames to be encoded. The shape of the input should be (num_vids, num_frames, *).
        :return: Encoded representation of each frame.
        """
        assert len(images.shape) == 5, "The input should be a sequence of video frames."
        # ensure the channels are first
        if images.shape[-1] == 3 and not images.shape[2] == 3:
            images = np.transpose(images, (0, 1, 4, 2, 3))
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_images = torch.tensor(batch_images, dtype=torch.float32).to(
                self.device
            )
            encoded_images = self._encode_image_batch(batch_images).cpu().numpy()
            if i == 0:
                encoded_images_all = encoded_images
            else:
                encoded_images_all = np.concatenate(
                    (encoded_images_all, encoded_images)
                )
        return encoded_images_all

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Since VLC just calls a server, we just return the same thing again
        """
        return images

    def send_to_server(self, texts: np.ndarray, videos: np.ndarray) -> float:
        """
        Sends video frames and text to the VLC server for reward calculation.
        :param video_frames: Numpy array containing all video frames.
        :param text: Corresponding textual description.
        :return: Computed reward.
        """
        # Image is (1, num_frames, 3, height, width)
        # We need to: pad num_frames to 12, and make sure height and width are 224
        videos = self.padding_video(videos[0], 12)

        # make sure height and width are 224
        # videos = torchvision.transforms.functional.center_crop(videos, (224, 224))
        videos = videos.numpy()

        # put the channels at the end
        videos = np.transpose(videos, (0, 2, 3, 1))

        # images should be unit8s
        videos = (videos * 255).astype(np.uint8)
        video_bytes = io.BytesIO()
        np.save(video_bytes, videos)
        video_encoded = base64.b64encode(video_bytes.getvalue()).decode("utf-8")

        request_data = {"video": video_encoded, "text": texts}
        request_data = pickle.dumps(request_data)

        response = requests.post(f"{self.server_url}/compute_reward", data=request_data)
        response.raise_for_status()

        reward = response.json()["reward"]

        print("Reward:", reward)
        return reward

    def calculate_rewards(
        self, video_frames: np.ndarray, text: str, camera_name: str = None
    ) -> float:
        """
        Calls the VLC server to compute the video-language reward.
        :param video_frames: Sequence of video frames.
        :param text: Corresponding textual description.
        :param camera_name: Name of the camera. Is ignored here
        :return: Computed reward.
        """
        return self.send_to_server(video_frames, text)

    def _calculate_reward_batch(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        pass

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
        return "vlc"

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        return 768  # for LIV

    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return 384  # for LIV

    def padding_video(self, video_frames, max_length):
        video_length = len(video_frames)
        if isinstance(video_frames, np.ndarray):
            video_frames = torch.tensor(video_frames)
        if video_length < max_length:
            # padding last frame
            padding_length = max_length - video_length
            # first_frame = video_frames[0].unsqueeze(0)
            last_frame = video_frames[-1].unsqueeze(0)
            # padding_frames = last_frame.repeat(padding_length, 1)
            padding_frames = last_frame.repeat(
                padding_length, *[1 for i in range(len(last_frame.shape) - 1)]
            )

            video_frames = torch.cat([video_frames, padding_frames], dim=0)
            # video_frames = th.cat([padding_frames, video_frames], dim=0)

        elif video_length > max_length:
            frame_idx = np.linspace(0, video_length - 1, max_length).astype(int)
            video_frames = video_frames[frame_idx]

        return video_frames


def read_video_as_frames(
    video_path: str, as_tensor: bool = False
) -> Union[List[np.ndarray], torch.Tensor]:
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
    video_frames = read_video_as_frames(
        "/scr/yusenluo/RoboCLIP/self_collected_vids/window_close/GT/1.mp4",
        as_tensor=False,
    )
    text = "closing the window"
    reward = reward_model.calculate_rewards(video_frames, text)
    print(reward)
