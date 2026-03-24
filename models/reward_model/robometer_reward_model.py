"""
Robometer reward model for no-action-chunk policy training.

Supports two modes:
  - Direct inference: loads Robometer model locally (requires robometer package)
  - HTTP server inference: calls a running Robometer eval server (no extra deps)

Usage:
  reward=robometer                              # direct mode
  reward=robometer reward_model.use_server=true # server mode

When this reward model is used, the policy still uses DINO embeddings for
observations (via image_encoder), but rewards are computed by Robometer
using raw image frames stored in an internal buffer.
"""

import numpy as np
import torch
from typing import List, Union

from models.reward_model.base_reward_model import BaseRewardModel


class RobometerRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: str = "aliangdw/Robometer-4B",
        device: str = "cuda",
        batch_size: int = 1,
        success_bonus: float = 64.0,
        reward_at_every_step: bool = True,
        max_frames: int = 4,
        use_server: bool = False,
        server_url: str = "http://localhost:8000",
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            success_bonus=success_bonus,
        )
        self.reward_at_every_step = reward_at_every_step
        self.max_frames = max_frames
        self.use_server = use_server
        self.server_url = server_url.rstrip("/")
        self.model_path = model_path

        # Frame buffer: stores raw frames (H, W, C) uint8
        self._frame_buffer: List[np.ndarray] = []
        # Task text: stored when encode_text is called
        self._task_text: str = ""

        if use_server:
            print(f"[RobometerRewardModel] Using HTTP server at {self.server_url}")
        else:
            print(f"[RobometerRewardModel] Loading model from {model_path} ...")
            self._load_robometer(model_path, device)
            print("[RobometerRewardModel] Model loaded successfully")

    def _load_robometer(self, model_path: str, device: str):
        """Load Robometer model and setup inference pipeline."""
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=model_path,
            device=self.device,
        )
        model.eval()

        self._robometer_model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._exp_config = exp_config
        self._batch_collator = setup_batch_collator(
            processor, tokenizer, exp_config, is_eval=True
        )

        # Determine discrete mode settings
        loss_config = getattr(exp_config, "loss", None)
        self._is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config
            else False
        )
        self._num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(exp_config.model, "progress_discrete_bins", 10)
        )

    # ------------------------------------------------------------------
    # Frame buffer management
    # ------------------------------------------------------------------
    def add_frame(self, image_for_model: np.ndarray):
        """
        Store a raw frame in the buffer. Called by the wrapper during step/reset.

        Args:
            image_for_model: shape (1, 1, H, W, C) uint8
        """
        frame = image_for_model[0, 0]  # (H, W, C)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._frame_buffer.append(frame.copy())

    def clear_frame_buffer(self):
        """Clear the frame buffer. Called by the wrapper on reset."""
        self._frame_buffer = []

    # ------------------------------------------------------------------
    # Override encode_text: store raw text + return MiniLM embedding
    # ------------------------------------------------------------------
    def encode_text(self, text: Union[str, List]) -> np.ndarray:
        if isinstance(text, list):
            self._task_text = text[0]
        else:
            self._task_text = text
        # Return MiniLM embedding for wrapper compatibility
        return super().encode_text(text)

    # ------------------------------------------------------------------
    # Override calculate_rewards: use buffered frames + task text
    # ------------------------------------------------------------------
    def calculate_rewards(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor],
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute progress score using Robometer.

        The encoded_texts and encoded_videos from the wrapper are DINO embeddings.
        Robometer ignores them and uses self._frame_buffer and self._task_text instead.
        """
        # Sync buffer length with wrapper's sequence length
        if isinstance(encoded_videos, torch.Tensor):
            T = encoded_videos.shape[1]
        else:
            T = encoded_videos.shape[1] if encoded_videos.ndim >= 2 else len(self._frame_buffer)

        if len(self._frame_buffer) > T:
            self._frame_buffer = self._frame_buffer[-T:]

        if len(self._frame_buffer) == 0:
            return np.array([0.0])

        frames = np.stack(self._frame_buffer, axis=0)  # (T, H, W, C) uint8

        if self.use_server:
            progress = self._infer_server(frames, self._task_text)
        else:
            progress = self._infer_local(frames, self._task_text)

        return np.array([progress])

    # ------------------------------------------------------------------
    # Direct inference (requires robometer package)
    # ------------------------------------------------------------------
    def _infer_local(self, frames: np.ndarray, task: str) -> float:
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        T = frames.shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[indices]
            T = self.max_frames
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = self._batch_collator([progress_sample])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)

        with torch.no_grad():
            results = compute_batch_outputs(
                self._robometer_model,
                self._tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=self._is_discrete,
                num_bins=self._num_bins,
            )

        progress_pred = results.get("progress_pred", [])
        if progress_pred and len(progress_pred) > 0:
            return float(progress_pred[0][-1])
        return 0.0

    # ------------------------------------------------------------------
    # HTTP server inference
    # ------------------------------------------------------------------
    def _infer_server(self, frames: np.ndarray, task: str) -> float:
        import requests
        import io
        import base64

        T = frames.shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[indices]

        buf = io.BytesIO()
        np.save(buf, frames)
        frames_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "frames_b64": frames_b64,
            "task": task,
            "sample_type": "progress",
        }

        try:
            resp = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            progress = result.get("progress", result.get("reward", 0.0))
            if isinstance(progress, list):
                return float(progress[-1])
            return float(progress)
        except Exception as e:
            print(f"[RobometerRewardModel] Server inference failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Abstract method stubs (not used directly)
    # ------------------------------------------------------------------
    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        return np.zeros((images.shape[0], images.shape[1], 2), dtype=np.float32)

    def _calculate_reward_batch(
        self, encoded_texts: np.ndarray, encoded_videos: np.ndarray
    ) -> np.ndarray:
        return np.array([0.0])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def img_output_dim(self) -> int:
        return 768  # Policy still uses DINO embeddings

    @property
    def text_output_dim(self) -> int:
        return 384

    @property
    def name(self) -> str:
        return "RobometerRewardModel"
