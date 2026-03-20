from models.reward_model import BaseRewardModel
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F
from models.encoders.dino_miniLM_encoder import Dino_miniLM_Encoder
import torch.nn as nn

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


class ReWiNDTransformer(nn.Module):
    """Matches the architecture used in rewind_valuemodel to load trained checkpoints."""
    def __init__(self, args, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.attention_mask = nn.Transformer.generate_square_subsequent_mask(args.max_length + 1).to('cuda')

    def forward(self, video_frames, text_embed, attention_mask=None):
        batch_size = video_frames.shape[0]

        video_embed = self.video_proj(video_frames)
        text_embed = self.text_proj(text_embed).unsqueeze(1)

        video_embed[:,0] += self.first_pos_embed

        sequence = torch.cat([text_embed, video_embed], dim=1)

        transformed = self.transformer(sequence, is_causal=True, mask=self.attention_mask)

        progress_preds = self.progress_head(transformed[:, 1:])

        return progress_preds


class RewindRewardModel(BaseRewardModel):
    def __init__(self, model_load_path: str, use_pca: bool, attention_heads: int, pca_model_dir: str = None, device: str = 'cuda', batch_size=64, reward_at_every_step: bool = False, success_bonus: int = 10, dino_batch_size: int = 32, max_num_frames_per_episode: int = 32, sum_reward: bool = False) -> None:
        """
        Initializes the RoboclipV2 reward model.
        :param model_load_path: Path to the model checkpoint.
        :param use_pca: Whether to use PCA for the video embeddings.
        :param attention_heads: Number of attention heads to use in the transformer model.
        :param pca_model_dir: Path to the PCA model checkpoint directory where the `pca_text.pkl` and `pca_video.pkl` files are located.
        :param device: Device to run the model on (default: 'cuda').
        :param batch_size: Batch size to use for encoding data (default: 64).
        """
        super().__init__(device, batch_size, success_bonus=success_bonus)
        self.use_pca = use_pca
        self.attention_heads = attention_heads
        self.model, self.model_args = self._load_model(model_load_path, pca_model_dir)
        self.reward_at_every_step = reward_at_every_step
        self.device = device
        self.dino_encoder = Dino_miniLM_Encoder(use_pca, device, dino_batch_size=dino_batch_size, max_num_frames_per_episode=max_num_frames_per_episode, batch_size=batch_size)
        self.max_num_frames_per_episode = max_num_frames_per_episode
        self.dino_batch_size = dino_batch_size
        self.sum_reward = sum_reward
    def _load_model(self, model_load_path: str, pca_model_path: str = None):
        video_dim = 768
        text_dim = 384
        model_dict = torch.load(model_load_path, map_location=self.device, weights_only=False)
        args = model_dict['args']
        model = ReWiNDTransformer(
                args=args,
                video_dim=video_dim,
                text_dim=text_dim,
                hidden_dim=512
            ).to(self.device)

        model.load_state_dict(model_dict['model_state_dict'])
        model.eval()
        return model, args


    def sample_embedding_frames(self, embeddings, num_frames = 32):
        total_frames = embeddings.shape[0]
        if total_frames > num_frames:
            index = np.linspace(0, total_frames-1, num_frames).astype(int)
            embeddings = embeddings[index]

        else:
            # padding last frame
            padding_num = num_frames - total_frames
            last_frame = embeddings[-1].unsqueeze(0)
            padding_frames = last_frame.repeat(padding_num, 1)
            embeddings = torch.cat([embeddings, padding_frames], dim=0)
        return embeddings


    def normalize_embeddings(self, embeddings, return_tensor=True):
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        if return_tensor:
            return normalized_embeddings
        else:
            return normalized_embeddings.detach().cpu().numpy()



    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        return self.dino_encoder.encode_text(text)

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_images, height, width, channels).
        :return: Encoded representation of each frame.
        """
        assert images.shape[0] == 1, "LIV doesn't support batch > 1"
        images = images.squeeze(0) # (128,3,224,224)
        indices = np.linspace(
                0,
                len(images) - 1,
                self.max_num_frames_per_episode,
                dtype=int,
            )
        sampled_images = torch.stack([images[i] for i in indices])
        # print(f"sampled_images.shape: {sampled_images.shape}") # (32, 3, 224, 224)
        return self.dino_encoder.encode_images(sampled_images.unsqueeze(0))
        

    def _calculate_reward_batch(self, encoded_texts: torch.Tensor, encoded_videos: torch.Tensor) -> torch.Tensor:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations. Shape: (batch_size, num_images, embedding_dim).
        :return: Reward values for each text-video pair.
        """
        # print(f"encoded_texts.shape: {encoded_texts.shape}, encoded_videos.shape: {encoded_videos.shape}")
        # TODO: add the processing for downsampling if needed @Yusen @Jiahui
        if getattr(self.model_args, 'normalize_embedding', False):
            encoded_videos = self.normalize_embeddings(encoded_videos)
        if getattr(self.model_args, 'subsample_video', True):
            processed_video_embedding = self.sample_embedding_frames(
                encoded_videos.squeeze(0), self.model_args.max_length
            ).unsqueeze(0)
        # print(f"processed_video_embedding.shape: {processed_video_embedding.shape}")
        pred_class = self.model(processed_video_embedding.float(), encoded_texts.float())
        pred_class = pred_class.squeeze(-1) # reward shape (1, 16, 1) -> (1, 16)
        if self.sum_reward:
            reward = torch.sum(pred_class, dim=1)
        else:
            reward = pred_class[:, -1]
        # print(f"reward before divisor: {reward}")
        return reward

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        if self.use_pca:
            return self.pca_video_model.components_.shape[0]
        return 768 # for LIV
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        if self.use_pca:
            return self.pca_text_model.components_.shape[0]
        return 384 # for LIV

    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'RewindRewardModel'

    def encode_text_for_policy(self, text: Union[str, List]) -> np.ndarray:
        """
        Encodes a text input into a representation for policy training.
        :param text: Text data to be encoded. If a list of strings is provided, it will be batch encoded.
        :return: Encoded representation of the text.
        """
        if isinstance(text, str):
            text = [text]
        return self.dino_encoder._encode_text_batch(text)