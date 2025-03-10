from models.reward_model.base_reward_model import BaseRewardModel
import torch
import numpy as np
from models.reward_model.s3dg import S3D
from typing import Union

class RoboclipRewardModel(BaseRewardModel):
    def __init__(self, device: str = "cuda", batch_size: int = 64, success_bonus: int = 10, model_load_path: str = "", reward_at_every_step: bool = False) -> None:
        super().__init__(device, batch_size, success_bonus=success_bonus)
        self.net = self.load_model()
        self.device = device
        self.reward_at_every_step = reward_at_every_step
    
    def padding_video(self, video_frames, max_length):
        num_vids, num_frames, c, h, w = video_frames.shape
        processed_videos = []

        for i in range(num_vids):
            frames = video_frames[i]  # shape: (num_frames, 3, H, W)
            length = frames.shape[0]

            if length < max_length:
                # 需要 padding
                padding_length = max_length - length
                first_frame = frames[0].unsqueeze(0)  # shape: (1, 3, H, W)
                # 重复第一帧 padding_length 次
                padding_frames = first_frame.repeat(padding_length, 1, 1, 1)  
                # 在第 0 维 (时间维) 拼接
                new_frames = torch.cat([padding_frames, frames], dim=0)  # (max_length, 3, H, W)

            elif length > max_length:
                # 超出长度则进行等间隔采样
                frame_idx = np.linspace(0, length - 1, max_length).astype(int)
                new_frames = frames[frame_idx]  # (max_length, 3, H, W)

            else:
                # 刚好等于 max_length
                new_frames = frames

            processed_videos.append(new_frames)

        # 在 batch 维度(第 0 维)把所有处理后的视频拼接起来
        padded_videos = torch.stack(processed_videos, dim=0)
        return padded_videos
    
    def load_model(self, model_load_path = 's3d_howto100m.pth'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = S3D('/home/yusenluo/RoboCLIP_offline/RoboCLIPv2/s3d_dict.npy', 512)
        net.load_state_dict(torch.load('/home/yusenluo/RoboCLIP_offline/RoboCLIPv2/s3d_howto100m.pth'))
        net = net.to(device)
        net.eval()
        return net

    def _encode_image_batch(self, images):
        # images = images[:, :, 240-112:240+112, 320-112:320+112, :3]
        images = self.padding_video(images, 32)
        images = images.permute(0, 2, 1, 3, 4)
        # images = images.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
        # print("images shape", images.shape) # (1,3,32,224,224)
        video_embeddings = self.net(images)["video_embedding"].to(self.device).float()
        return video_embeddings

    def _encode_text_batch(self, text):
        text_embeddings = self.net.text_module(text)["text_embedding"].to(self.device).float()
        return text_embeddings
    
    def _calculate_reward_batch(self, text_embeddings, video_embeddings):
        text_embeddings = text_embeddings.squeeze(0)
        # print("video_embeddings shape", video_embeddings.shape)
        return torch.matmul(video_embeddings, text_embeddings.t())[0].detach().cpu().numpy()

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        return 512 # for S3D
    
    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return 512 # for S3D
    
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'S3DRewardModel'
    
