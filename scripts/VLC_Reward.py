import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Union
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize 
from torchvision.transforms import functional as F
from video_language_critic.reward import RewardCalculator
# from paths import *
import os
import matplotlib.pyplot as plt
import cv2
import imageio
# from animate_utils import animate_reversed_incremental, animate_reversed_incremental_compare

def read_video_as_frames(video_path: str, as_tensor: bool = False) -> Union[List[np.ndarray], torch.Tensor]:
    """
    从给定的 GIF 或 MP4 等视频文件路径中，读取所有帧并返回帧列表或张量。
    每个帧为 shape = (H, W, 3) 的 np.ndarray（RGB 或 BGR）。

    参数:
    ----
    video_path : str
        本地视频文件路径，比如 '/path/to/video.gif' 或 '/path/to/video.mp4'
    as_tensor : bool, default=False
        如果为 True，则返回一个形状为 [num_frames, H, W, 3] 的张量 (torch.Tensor)；
        否则返回帧列表 (List[np.ndarray])。

    返回:
    ----
    Union[List[np.ndarray], torch.Tensor]
        如果 as_tensor=False，返回按顺序存储每一帧的列表；
        如果 as_tensor=True，返回形状为 [num_frames, H, W, 3] 的张量。
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = []
    ext = os.path.splitext(video_path)[1].lower()

    if ext == ".gif":
        # 使用 imageio 读取 GIF
        reader = imageio.get_reader(video_path)
        for frame in reader:
            # imageio 读取的 GIF 帧一般是 RGB 格式
            frames.append(frame)  # shape=(H, W, 3), dtype=uint8
        reader.close()
    else:
        # 使用 OpenCV 读取 (mp4/avi/mov 等)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV 读取到的是 BGR 通道顺序，需要转为 RGB 通道
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)  # shape=(H, W, 3), dtype=uint8
        
        cap.release()

    # 如果 as_tensor 为 True，将帧转换为张量
    if as_tensor:
        # 将帧列表转换为 NumPy 数组，然后转换为 PyTorch 张量
        frames_array = np.stack(frames, axis=0)  # shape=(num_frames, H, W, 3)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)  # shape=(num_frames, 3, H, W)
        return frames_tensor

    return frames


def tensor_to_frame_list(frames_tensor: torch.Tensor) -> List[np.ndarray]:
    """
    将视频帧张量还原为帧列表。

    参数:
    ----
    frames_tensor : torch.Tensor
        视频帧张量，形状为 [num_frames, 3, H, W]。

    返回:
    ----
    List[np.ndarray]
        按顺序存储每一帧的列表；每一帧是形状为 (H, W, 3) 的 NumPy 数组。
    """
    # 调整张量形状为 (num_frames, H, W, 3)
    frames_array = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()  # 转为 NumPy 数组
    # 将 NumPy 数组切片为帧列表
    frames_list = [frames_array[i] for i in range(frames_array.shape[0])]
    return frames_list


def load_vlc_args(vlc_ckpt: str):
    """
    与原脚本一致的加载逻辑
    """
    init_model_path = os.path.join("/home/jzhang96/RoboCLIPv2/scripts", vlc_ckpt)
    vlc_args_path = os.path.join(init_model_path + '_config.pkl')
    with open(vlc_args_path, 'rb') as f:
        vlc_args = pickle.load(f)['args']
    vlc_args.init_model = "/home/jzhang96/RoboCLIPv2/scripts/pytorch_model.bin.20" #"/scr/yusenluo/video_language_critic/experiments/mw50_training/pytorch_model.bin.20" 
    vlc_args.resume_from_latest = False
    return vlc_args


class SingleVideoVLCRewardCalculator:
    """
    一次性传入一整段视频(帧列表)，输出 Video-Language Critic (VLC) Reward。
    与原脚本的 RewardCalculator、transform、推理逻辑保持一致。
    """

    def __init__(
        self,
        vlc_args,
        env_id: str,
        caption_text: str = None,
        stretch_partial_videos: bool = False,
        device: str = "cuda",
    ):
        """
        参数:
        ----
        vlc_args: 与原脚本保持一致，通常通过 load_vlc_args 得到
        env_id: 原脚本中可能会根据 env_id 去找 caption。这里直接保留，用来做名称
        caption_text: 若您想自定义文本描述, 可以传; 若为空, 则可在外部做加载
        stretch_partial_videos: 是否在帧数不足 max_frames 时也做插帧到 max_frames
        device: "cuda:0" 或 "cpu"
        """

        self.vlc_args = vlc_args
        self.env_id = env_id
        self.stretch_partial_videos = stretch_partial_videos
        self.device = device

        # 与原脚本相同: 构造 RewardCalculator
        self.reward_model = RewardCalculator(args=vlc_args)
        self.reward_model.model.eval()
        self.reward_model.model.to(device)

        # 原脚本中, 可能从 CAPTION_PATH / raw-captions.pkl 中拿到 caption
        # 这里若有需要可以重复原逻辑:
        if caption_text is None:
            # 简单示例: 也可以自动从 descriptions 里挑
            with open(f'{CAPTION_PATH}/raw-captions.pkl', 'rb') as f:
                descriptions = pickle.load(f)
            # 假设找到第一个包含 env_id 的条目
            # 在原脚本中: [v for k, v in descriptions.items() if env_id in k][0][0]
            # 这里简单示例:

            self.caption_text = [
                v for k, v in descriptions.items() if env_id in k
            ][0][0]
            self.caption_text = " ".join(self.caption_text)
            #self.caption_text = [v for k, v in descriptions.items() if 'success_videos__' + env_id in k][0][0] 
        else:
            self.caption_text = caption_text
        
        print("Caption text:", self.caption_text)

        # 同原脚本: transform
        self.transform = Compose(
            [
                Resize(224, interpolation=Image.Resampling.BICUBIC),
                CenterCrop(224),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        # 解析出 max_frames, frame_indices_to_use 等
        self.max_frames = getattr(vlc_args, "max_frames", 12)
        self.frame_indices_to_use = getattr(vlc_args, "frame_indices_to_use", None)

        # 与原脚本相同: 先将文本处理成 pairs_text, pairs_mask, pairs_segment
        (
            self.pairs_text,
            self.pairs_mask,
            self.pairs_segment,
            _
        ) = self.reward_model.dataloader._get_text(
            video_id=0,
            caption=self.caption_text
        )
        # 转为 tensor
        self.pairs_text = torch.tensor(self.pairs_text).to(self.device)
        self.pairs_mask = torch.tensor(self.pairs_mask).to(self.device)
        self.pairs_segment = torch.tensor(self.pairs_segment).to(self.device)

    def set_text(self, caption_text: str):
        """
        设置新的文本描述，用于计算 VLM Reward。
        """
        self.caption_text = caption_text
        # 重新构建 pairs_text, pairs_mask, pairs_segment
        (
            self.pairs_text,
            self.pairs_mask,
            self.pairs_segment,
            _
        ) = self.reward_model.dataloader._get_text(
            video_id=0,
            caption=self.caption_text
        )
        # 转为 tensor
        self.pairs_text = torch.tensor(self.pairs_text).to(self.device)
        self.pairs_mask = torch.tensor(self.pairs_mask).to(self.device)
        self.pairs_segment = torch.tensor(self.pairs_segment).to(self.device)

    def transform_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        对一批原生帧进行 transform，返回形状 [num_frames, 3, 224, 224] 的 Torch 张量。
        frames: shape = [num_frames, H, W, 3] (RGB) (或其他顺序，看情况)
        """
        # 假设 frames 是一个 list / np.array of shape [num_frames, H, W, 3]
        transformed_list = []
        for i, frame in enumerate(frames):
            # 转成 PIL 再 transform
            # (这里的 self.transform 通常包含 Resize, CenterCrop, ToTensor, Normalize)
            pil_img = Image.fromarray(frame.astype(np.uint8))  # => PIL
            img_t = self.transform(pil_img)  # => shape [3,224,224]
            transformed_list.append(img_t)
        # 拼成一个整体张量
        video_tensor = torch.stack(transformed_list, dim=0)  # => [num_frames, 3,224,224]
        print("Video tensor shape:", video_tensor.shape)
        return video_tensor
    
    def padding_frames(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        对 video_tensor 做抽帧，如果帧数 >= max_frames 或 stretch_partial_videos=True，就用 np.linspace；
        否则直接 arange。返回抽帧后的子集 (shape [selected_len, 3,224,224])。
        """
        num_frames = video_tensor.shape[0]
        if num_frames >= self.max_frames or self.stretch_partial_videos:
            float_indices = np.linspace(0, num_frames - 1, self.max_frames)
            indices = np.round(float_indices).astype(int)
        else:
            indices = np.arange(num_frames)

        if self.frame_indices_to_use is not None and len(indices) > len(self.frame_indices_to_use):
            indices = indices[self.frame_indices_to_use]

        # 根据抽帧后的 indices 取子集
        sampled_video_tensor = video_tensor[indices]  # shape [selected_len, 3,224,224]
        return sampled_video_tensor

    def build_batch_and_mask(self, sampled_video_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        将抽帧后的视频张量填充到 [1,1, max_frames,1,3,224,224] 并构建 mask => [1, max_frames].
        返回 (batch_tensor, video_mask, real_count).
        """
        selected_len = sampled_video_tensor.shape[0]
        # 构造 batch => shape = [B=1, 1, max_frames, 1, 3,224,224]
        batch_tensor = torch.zeros(
            (1, 1, self.max_frames, 1, 3, 224, 224), dtype=torch.float32, device=self.device
        )
        video_mask = torch.zeros((1, self.max_frames), dtype=torch.int64, device=self.device)

        real_count = min(selected_len, self.max_frames)
        video_mask[0, :real_count] = 1

        # 将实际帧放到前 real_count 里
        for i in range(real_count):
            batch_tensor[0, 0, i, 0] = sampled_video_tensor[i]

        return batch_tensor, video_mask, real_count

    def compute_video_reward(self, processed_video_tensor: torch.Tensor, last_frame_reward_only: bool = True) -> float:
        """
        一次性传入一整段视频(张量)，返回对应的 VLM Reward。
        与原脚本的帧处理、linspace、frame_indices_to_use、mask 等逻辑保持一致。
        
        参数:
        ----
        video_tensor : torch.Tensor
            视频帧张量，形状为 [num_frames, 3, H, W]。
        last_frame_reward_only : bool, default=False
            如果为 True，则返回最后一帧的奖励；否则返回完整数组。

        返回:
        ----
        float
            对应视频的 VLC Reward。
        """
        processed_video_tensor = self.padding_frames(processed_video_tensor)
        assert processed_video_tensor.shape[0] == self.max_frames, f"Expected {self.max_frames} frames, got {processed_video_tensor.shape[0]}"
        # print("Padded video tensor shape:", processed_video_tensor.shape)
        processed_video_tensor, video_mask, real_count = self.build_batch_and_mask(processed_video_tensor)
        # 推理: 与原脚本一致
        with torch.no_grad():
            expanded_mask = video_mask.unsqueeze(1)  # => [1, 1, T]
            # print("Expanded mask shape:", expanded_mask.shape)
            a, b = self.reward_model.model.get_sequence_visual_output(
                self.pairs_text,
                self.pairs_mask,
                self.pairs_segment,
                processed_video_tensor,
                expanded_mask
            )
            scores = self.reward_model.model.get_similarity_logits(
                a,
                b,
                self.pairs_text,
                expanded_mask,
                loose_type=self.reward_model.model.loose_type
            )[0]

        # 若 scores 是 [B,T], 就取最后帧, 与原脚本一致
        print("Scores shape:", scores.shape)
        if len(scores.shape) > 1:
            final_idx = real_count - 1  # 最后有效帧
            if last_frame_reward_only:
                vlm_reward = float(scores[0, 0, final_idx].item())
            else:
                vlm_reward = scores[0, 0, :real_count].detach().cpu().numpy()
        else:
            vlm_reward = float(scores.item())

        return vlm_reward

    

if __name__ == "__main__":
    # video_path = "/home/jzhang96/RoboCLIPv2/liv_train_final/eval_tasks_v2/button-press-wall-v2/close_succ/1.gif"  # 或 .gif /scr/yusenluo/RoboCLIP/self_collected_vids/button_press/GT/2.gif
    #video_frames = read_video_as_frames(video_path)
    # video_frames = read_video_as_frames(video_path, as_tensor=False)


    video_frames = np.random.randint(0, 255, (12, 224, 224, 3), dtype=np.uint8) # 这里12帧只是举例，可以换成任意帧数，因为会自动padding到12帧
    # 2) 载入 VLM args
    vlc_ckpt_name = "ckpt_mw40_retrank33_tigt_negonly_a_rf_1__pytorch_model.bin.20" 
    ##记得scp一下 "/home/yusenluo/vlc_rl/VLC_trained_with_mw19/pytorch_model.bin.20"， 然后改成你的路径
    # 还有这个 /home/yusenluo/vlc_rl/vlc_ckpts/vlc_ckpts/ckpt_mw40_retrank33_tigt_negonly_a_rf_1__pytorch_model.bin.20_config.pkl
    vlc_args = load_vlc_args(vlc_ckpt_name) 

    # 3) 初始化 RewardCalculator (或我们示例的 SingleVideoVLCRewardCalculator)
    single_video_calc = SingleVideoVLCRewardCalculator(
        vlc_args=vlc_args,
        caption_text="press the button", # 初始化时，caption_text可以随便写，只要set_text时能换就行
        env_id="button-press-v2", # 有caption_text的话，env_id可以随便写
        stretch_partial_videos=True,
        device="cuda"
    )

    single_video_calc.set_text("press the button") # 这里能换text，不用重新初始化single_video_calc

    # 4) 一次性计算该段视频的 VLM Reward
    vlm_reward = single_video_calc.compute_video_reward(single_video_calc.transform_frames(video_frames), last_frame_reward_only=True)
    print("Computed VLM reward:", vlm_reward)


    