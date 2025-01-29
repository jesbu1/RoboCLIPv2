from reward_model.base_reward_model import BaseRewardModel
import torch
import numpy as np
from s3dg import S3D
from typing import Union

class RoboclipRewardModel(BaseRewardModel):
    def __init__(self, device: str = "cuda", batch_size: int = 64, success_bonus: int = 10) -> None:
        self.model = self.load_model()
        self.device = device
    
    def padding_video(video_frames, max_length):
        video_length = len(video_frames)
        if type(video_frames) == np.ndarray:
            video_frames = torch.tensor(video_frames)
        if video_length < max_length:
            # padding first frame
            padding_length = max_length - video_length
            first_frame = video_frames[0].unsqueeze(0)
            padding_frames = first_frame.repeat(padding_length, 1)
            video_frames = torch.cat([padding_frames, video_frames], dim=0)
        
        elif video_length > max_length:
            frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
            video_frames = video_frames[frame_idx]

        return video_frames
    
    def load_model(self, model_load_path = '../s3d_howto100m.pth'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = S3D('../s3d_dict.npy', 512)
        net.load_state_dict(torch.load('../s3d_howto100m.pth'))
        net = net.to(device)
        net.eval()
        return net

    def encode_images(self, images):
        images = images[:, 240-112:240+112, 320-112:320+112, :]
        images = self.padding_video(images, 32)
        images = images.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
        video_embeddings = self.net(images)["video_embedding"].to(self.device).float()
        return video_embeddings

    def encode_text(self, text):
        text_embeddings = self.net.text_module(text)["text_embedding"].to(self.device).float()
        return text_embeddings
    
    def calculate_rewards(self, text_embeddings: Union[np.ndarray, torch.Tensor], video_embeddings: Union[np.ndarray, torch.Tensor]):
        return torch.matmul(video_embeddings, text_embeddings.t())[0].detach().cpu().numpy()