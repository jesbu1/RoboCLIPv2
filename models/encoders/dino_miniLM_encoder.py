from models.encoders.base_encoder import BaseEncoder
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T



class Dino_miniLM_Encoder(BaseEncoder):
    def __init__(self, use_pca: bool, device: str = 'cuda', dino_batch_size=32, max_num_frames_per_episode=32, batch_size=128):
        """
        Initializes the DINO-MiniLM encoder.
        :param use_pca: Whether to use PCA for the video embeddings.
        :param device: Device to run the model on (default: 'cuda').
        :param batch_size: Batch size to use for encoding data (default: 128).
        """
        super().__init__(device, batch_size)
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
            device
        )


        self.use_pca = use_pca
        self.DINO_BATCH_SIZE = dino_batch_size
        self.MAX_NUM_FRAMES_PER_EPISODE = max_num_frames_per_episode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False)
        self.dinov2_vits14 = self.dinov2_vits14.to(self.device)
        self.dino_transform_image = T.Compose(
            [T.ToTensor(), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
        )
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
            self.device
        )


    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        encoded_input = self.minilm_tokenizer(
            text, padding=False, truncation=True, return_tensors="pt"
        ).to(self.device)

        model_output = self.minilm_model(**encoded_input)
        minlm_task_embedding = (
            self.mean_pooling(model_output, encoded_input["attention_mask"])
            .cpu()
            .detach()
            .numpy()
        )

        text_embeddings = np.concatenate(minlm_task_embedding, axis=0)
        return text_embeddings

    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        # videos.shape = (128, 224, 224, 3), dtype=uint8
        # Convert input from (1,3,480,640) to (1,480,640,3)
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        # print(f"images.shape before transpose: {images.shape}") # (1,1,3,480,640)
        if images.shape[2] == 3:
        # Adjust channel order from (1,3,480,640) to (1,480,640,3)
            images = np.transpose(images, (0, 1, 3, 4, 2)).squeeze(0)
        # print(f"images.shape after transpose: {images.shape}") # (1,480,640,3)
        
        # Ensure data type is uint8, range 0-255
        # Note: BaseEncoder.encode_images converts uint8 [0,255] to float32 [0,255],
        # so we just need to cast back to uint8, NOT multiply by 255.
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)
        # print(images)
        assert images.dtype == np.uint8, "must be uint8"
        assert np.min(images) >= 0 and np.max(images) <= 255, "must be between 0 and 255"
        assert not (np.max(images) <= 1 and np.min(images) >= 0), "must not be between 0 and 1"
        
        # print(f"images.shape: {images.shape}")
        # Process all images directly, without downsampling
        with torch.inference_mode():
            episode_images_dino = [self.dino_load_image(img) for img in images]
            # Batch processing
            episode_images_dino = [
                torch.concatenate(episode_images_dino[i : i + self.DINO_BATCH_SIZE])
                for i in range(0, len(episode_images_dino), self.DINO_BATCH_SIZE)
            ]
            embedding_list = []
            for batch in episode_images_dino:
                # print(f"batch.shape: {batch.shape}") # (1,3,224,224)
                episode_image_embeddings = (
                    self.dinov2_vits14(batch.to(self.device))
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                embedding_list.append(episode_image_embeddings)
            episode_image_embeddings = np.concatenate(embedding_list)

        return episode_image_embeddings

    def dino_load_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Load an image and return a tensor that can be used as an input to DINOv2.
        """
        # Ensure image is uint8 type
        if img.dtype != np.uint8:
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
                
        # Ensure image is 3-channel RGB
        if img.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, got {img.shape[-1]}")
            
        img = Image.fromarray(img)

        transformed_img = self.dino_transform_image(img)[:3].unsqueeze(0)

        return transformed_img

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        if self.use_pca:
            return self.pca_video_model.components_.shape[0]
        return 768 # for Dino
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        if self.use_pca:
            return self.pca_text_model.components_.shape[0]
        return 384 # for MiniLM
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'dino_miniLM_Encoder'