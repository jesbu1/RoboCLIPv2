"""
TOPReward reward model for no-action-chunk policy training.

The policy still receives DINO features from the shared image encoder. The reward
model stores raw rendered frames and queries a TOPReward vLLM server for progress.
"""

import base64
import fcntl
import io
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from models.reward_model.base_reward_model import BaseRewardModel


class TOPRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        batch_size: int = 1,
        success_bonus: float = 0.0,
        reward_at_every_step: bool = True,
        max_frames: int = 4,
        server_url: str = "http://localhost:8000",
        request_timeout: float = 900.0,
        request_retries: int = 3,
        lock_path: str = "",
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            success_bonus=success_bonus,
        )
        self.model_name = model_name
        self.reward_at_every_step = reward_at_every_step
        self.max_frames = max_frames
        self.server_url = server_url.rstrip("/")
        self.request_timeout = request_timeout
        self.request_retries = max(int(request_retries), 1)
        self.lock_path = lock_path
        self._frame_buffer: List[np.ndarray] = []
        self._task_text = ""

    def _center_crop_frame(self, frame: np.ndarray, crop_size: int = 224) -> np.ndarray:
        if frame.ndim != 3:
            return frame
        frame = frame[..., :3]
        height, width = frame.shape[:2]
        crop_height = min(crop_size, height)
        crop_width = min(crop_size, width)
        top = max((height - crop_height) // 2, 0)
        left = max((width - crop_width) // 2, 0)
        return np.ascontiguousarray(
            frame[top : top + crop_height, left : left + crop_width]
        )

    def add_frame(self, image_for_model: np.ndarray):
        frame = image_for_model[0, 0]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3:
            frame = self._center_crop_frame(frame)
        self._frame_buffer.append(frame.copy())

    def clear_frame_buffer(self):
        self._frame_buffer = []

    def _subsample_frames(self) -> List[np.ndarray]:
        if len(self._frame_buffer) <= self.max_frames:
            return self._frame_buffer
        indices = np.linspace(0, len(self._frame_buffer) - 1, self.max_frames, dtype=int)
        return [self._frame_buffer[idx] for idx in indices]

    def _frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        b64_frames = []
        for frame in frames:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG")
            b64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return b64_frames

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        with torch.no_grad():
            encoded_input = self.tokenizer(
                text, padding=False, truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded_input)
            from models.reward_model.base_reward_model import mean_pooling

            text_embeddings = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .numpy()
            )
        return text_embeddings

    def encode_text(self, text: Union[str, List]) -> np.ndarray:
        self._task_text = text[0] if isinstance(text, list) else text
        return super().encode_text(text)

    def calculate_rewards(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor, None] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if len(self._frame_buffer) == 0:
            return np.array([0.0])
        return np.array([self._infer_server()])

    def _infer_server(self) -> float:
        import requests

        frames_b64 = self._frames_to_base64(self._subsample_frames())
        prompt_text = (
            "The above video shows a robot manipulation trajectory "
            "that completes the following task: "
        )
        instruction_suffix = (
            f"{self._task_text} Decide whether the above statement is True or not. "
            "The answer is:"
        )

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
            for b64 in frames_b64
        ]
        content.append({"type": "text", "text": f"{prompt_text}{instruction_suffix}"})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }

        for attempt in range(1, self.request_retries + 1):
            lock_file = None
            try:
                if self.lock_path:
                    lock_file = open(self.lock_path, "w")
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                resp = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                print(
                    f"[TOPRewardModel] Server inference failed on attempt "
                    f"{attempt}/{self.request_retries}: {exc}"
                )
                if attempt == self.request_retries:
                    return -10.0
                continue
            finally:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()

            logprobs = result["choices"][0]["logprobs"]["content"]
            if logprobs:
                first_token = logprobs[0]
                for entry in first_token.get("top_logprobs", []):
                    if entry["token"].strip().lower() == "true":
                        return float(entry["logprob"])
                if first_token["token"].strip().lower() == "true":
                    return float(first_token["logprob"])
            return -10.0

        return -10.0

    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        return np.zeros((images.shape[0], images.shape[1], 2), dtype=np.float32)

    def _calculate_reward_batch(
        self,
        encoded_texts: torch.Tensor,
        encoded_videos: torch.Tensor,
    ) -> np.ndarray:
        return np.array([0.0])

    @property
    def img_output_dim(self) -> int:
        return 768

    @property
    def text_output_dim(self) -> int:
        return 384

    @property
    def name(self) -> str:
        return "TOPRewardModel"
