import torch
from PIL import Image
#from dataloader_liv import video_collate_fn, LivVideoDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
import wandb
from tqdm import tqdm
import h5py
# from eval_video_self_attention_pca import plot_progress, plot_progress_corr, plot_videos
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from models import MultiHeadAttentionModel, MultiHeadAttentionSubtraction, MultiHeadAttentionConcatenation
#from eval_utils import plot_progress, plot_progress_class, plot_videos, plot_videos_class
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch as th
import io
from clip_utils import load_model, embedding_text, embedding_image
#from confusion_matrix import plot_confusion_matrix_pca, plot_confusion_matrix_pca_class


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--text_string', type=str, default='opening door')
    parser.add_argument('--pca', action="store_true")
    parser.add_argument('--attention_heads', type=int, default=4)

    args = parser.parse_args()
    return args


def animate_reversed_incremental(frames_tensor, incremental_rewards, fps=15):
    """
    将「帧序列 + 对应 incremental reward」动态可视化。
    左半部分：视频帧
    右半部分：到当前帧为止的 reward 曲线

    参数：
    - frames_tensor: shape [N, C, H, W] 的 torch.Tensor，值范围约在 [0,1] (一般由 ToTensor 转换)
    - incremental_rewards: Python list 或 numpy array，长度为 N，对应每个帧的 reward
    - fps: GIF 的帧率
    返回：
    - gif_buffer: 包含GIF的 BytesIO 缓冲，可直接用于wandb等日志系统
    """

    # ----------------------------
    # 1. 先把 frames_tensor 转成可在 matplotlib 中显示的格式
    # ----------------------------
    # (N, C, H, W) -> (N, H, W, C)，同时将 [0,1] 范围转换到 [0,255] 并变为 uint8
    frames_np = frames_tensor.cpu().numpy()  # 如果在GPU上，需要先转到CPU
    frames_np = (frames_np * 255).astype(np.uint8)  # 假设原先是 [0,1]
    frames_np = np.transpose(frames_np, (0, 2, 3, 1))  # 变为 (N, H, W, C)

    # ----------------------------
    # 2. 准备绘制
    # ----------------------------
    n = len(frames_np)  # 总帧数
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 左侧：显示帧
    image_plot = ax1.imshow(frames_np[0])
    ax1.set_title('Video Frames')
    ax1.axis('off')  # 不显示坐标

    # 右侧：显示增量 reward 曲线
    ax2.set_title('Incremental Rewards')
    ax2.set_xlim(0, n - 1)  # x 从 0 到 n-1
    # 根据 reward 的最小值和最大值，留一点上下边距
    y_min, y_max = min(incremental_rewards), max(incremental_rewards)
    y_range = y_max - y_min if y_max != y_min else 1
    ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    line_plot, = ax2.plot([], [], lw=2, color='blue')
    scat = ax2.scatter([], [], color='red', zorder=5)

    # 用于做 GIF 的缓冲
    gif_buffer = io.BytesIO()
    images = []

    # ----------------------------
    # 3. 动态更新并收集每一帧的图像
    # ----------------------------
    for frame_idx in range(n):
        # 更新左侧帧
        image_plot.set_array(frames_np[frame_idx])

        # 更新右侧曲线(0 ~ frame_idx)
        line_plot.set_data(np.arange(frame_idx + 1), incremental_rewards[:frame_idx + 1])
        scat.set_offsets(np.array([[frame_idx, incremental_rewards[frame_idx]]]))

        # 重新绘制
        fig.canvas.draw()

        # 将当前 figure 保存到 np.array 再转成 PIL Image
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = img_array.reshape((h, w, 3))
        images.append(Image.fromarray(img_array))

    # ----------------------------
    # 4. 将图像序列保存为 GIF
    # ----------------------------
    images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
    gif_buffer.seek(0)

    plt.close(fig)  # 关闭 figure，释放资源
    return gif_buffer



def save_reversed_video(reversed_frames_tensor, output_path="reversed_video.mp4", fps=30):
    # reversed_frames_tensor: (N, C, H, W), C=3, RGB格式，值范围 [0,1]

    # 将tensor移动到CPU，并转换为numpy数组
    reversed_frames = reversed_frames_tensor.cpu().numpy()  # shape: (N, C, H, W)

    # 转置为 (N, H, W, C) 以匹配OpenCV预期的图像格式
    reversed_frames = reversed_frames.transpose(0, 2, 3, 1)  # (N, H, W, C)

    # 转换到[0,255]的uint8格式
    reversed_frames = (reversed_frames * 255).astype(np.uint8)

    # OpenCV使用BGR格式，所以需要转换
    reversed_frames_bgr = reversed_frames[..., ::-1]

    # 获取帧的高度和宽度
    height, width = reversed_frames_bgr.shape[1], reversed_frames_bgr.shape[2]

    # 定义VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入帧
    for frame in reversed_frames_bgr:
        out.write(frame)

    out.release()
    print(f"Saved reversed video to {output_path}")


class VideoRewardEvaluator:
    def __init__(self, args, model, processor, transform_model, target_embedding, pca_video_model=None):
        self.model = model
        self.args = args
        self.processor = processor
        self.transform_model = transform_model.cuda()
        self.target_embedding = target_embedding.cuda()
        self.pca_video_model = pca_video_model

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def padding_video(self, frames, max_length=32):
        total_frames = frames.shape[0]  # 获取帧数

        if total_frames > max_length:
            # 使用均匀抽样
            indices = torch.linspace(0, total_frames - 1, max_length).long()  # 均匀采样索引
            frames = frames[indices]
        else:
            # 对第一个帧进行填充
            padding_num = max_length - total_frames
            first_frame = frames[0].unsqueeze(0)  # 添加维度，确保填充形状一致
            padding_frames = first_frame.repeat(padding_num, 1, 1, 1)  # 重复填充
            frames = torch.cat([padding_frames, frames], dim=0)  # 拼接填充帧和原始帧

        return frames


    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转成RGB格式，方便后续处理
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()

        # 如果至少有1帧，则保存第一帧和最后一帧查看
        # if len(frames) > 0:
        #     first_frame = frames[0]
        #     last_frame = frames[-1]

        #     # 转回BGR以使用cv2.imwrite正确保存
        #     first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        #     last_frame_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)

        #     cv2.imwrite("first_frame.jpg", first_frame_bgr)
        #     cv2.imwrite("last_frame.jpg", last_frame_bgr)
        #     print("Saved first_frame.jpg and last_frame.jpg")

        return frames


    def preprocess_frames(self, frames):
        processed_frames = [
            self.transform(Image.fromarray(frame)) for frame in frames
        ]
        processed_frames = [
                    frame[
                        :3,  
                        (frame.shape[1] - 224) // 2 : (frame.shape[1] + 224) // 2,
                        (frame.shape[2] - 224) // 2 : (frame.shape[2] + 224) // 2,
                    ]
                    for frame in processed_frames
                ]
        return th.stack(processed_frames)

    def compute_reward(self, video_path):
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        length = frames_tensor.shape[0]

        # 准备四种不同的帧子集策略
        # 1. 所有帧
        frames_all = frames_tensor

        # 2. 前半段帧
        frames_front_half = frames_tensor[: length // 2]
        #frames_front_half = self.padding_video(frames_front_half)

        # 3. 后半段帧
        frames_back_half = frames_tensor[length // 2 :]

        # 4. 均匀抽取18帧（包括第一帧和最后一帧）
        # 如果视频长度小于18帧，则 linspace 会产生重复索引，这种情况下不会有问题
        # 但你也可以根据需要进行特殊处理
        #indices = np.linspace(0, length - 1, num=32, dtype=int)
        frames_uniform_32 = self.padding_video(frames_tensor)

        print("Frames Tensor Shape (all):", frames_all.shape)
        print("Frames Tensor Shape (front half):", frames_front_half.shape)
        print("Frames Tensor Shape (back half):", frames_back_half.shape)
        print("Frames Tensor Shape (uniform 32):", frames_uniform_32.shape)

        # 定义一个内部函数，用于对给定的 frames 子集计算 reward
        def compute_subreward(sub_frames):
            with th.no_grad():
                # 将子集嵌入
                video_embeddings = embedding_image(self.model, self.processor, sub_frames).cuda()
                if self.pca_video_model:
                    video_embeddings = self.pca_video_model.transform(video_embeddings.cpu().numpy())
                    video_embeddings = th.from_numpy(video_embeddings).float().cuda()

            video_embeddings = video_embeddings.view(1, -1, video_embeddings.shape[-1]).float()
            # 利用 transform_model 和 target_embedding 计算 reward
            sub_reward = self.transform_model(video_embeddings, None, self.target_embedding).item()
            return sub_reward

        reward_all = compute_subreward(frames_all)
        reward_front_half = compute_subreward(frames_front_half)
        reward_back_half = compute_subreward(frames_back_half)
        reward_uniform_32 = compute_subreward(frames_uniform_32)

        print(f"Reward (All Frames): {reward_all}")
        print(f"Reward (Front Half): {reward_front_half}")
        print(f"Reward (Back Half): {reward_back_half}")
        print(f"Reward (Uniform 32): {reward_uniform_32}")

        return {
            'all': reward_all,
            'front_half': reward_front_half,
            'back_half': reward_back_half,
            'uniform_32': reward_uniform_32
        }
    
    def test_incremental_frames_reward(self, video_path, output_path="incremental_reward_plot.png", gif_path="incremental.gif"):
        """
        从第一帧开始，依次增加帧数，直到使用全部帧数，计算每个阶段的reward，并保存一张折线图。
        """
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        frames_tensor = self.padding_video(frames_tensor)
        length = frames_tensor.shape[0]

        incremental_rewards = []

        # 逐帧增加
        for i in range(1, length + 1):
            sub_frames = frames_tensor[:i]
            with th.no_grad():
                video_embeddings = embedding_image(self.model, self.processor, sub_frames).cuda()
                if self.pca_video_model:
                    video_embeddings = self.pca_video_model.transform(video_embeddings.cpu().numpy())
                    video_embeddings = th.from_numpy(video_embeddings).float().cuda()

            video_embeddings = video_embeddings.view(1, -1, video_embeddings.shape[-1]).float()
            sub_reward = self.transform_model(video_embeddings, None, self.target_embedding).item()
            incremental_rewards.append(sub_reward)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, length + 1), incremental_rewards, marker='o')
        plt.title('Reward vs Number of Frames')
        plt.xlabel('Number of Frames Used')
        plt.ylabel('Reward')
        plt.grid(True)

        # 保存图片
        plt.savefig(output_path)
        plt.close()

        gif_buffer = animate_reversed_incremental(frames_tensor, incremental_rewards, fps=15)

        # 如果想要将 gif_buffer 存到本地文件:
        with open(gif_path, 'wb') as f:
            f.write(gif_buffer.getvalue())
        print(f"Saved incremental reversed GIF to {gif_path}")

        return incremental_rewards
    
    def test_incremental_frames_with_reversed(self, video_path, output_path="incremental_reward_reversed_plot.png", gif_path="reversed_incremental.gif"):
        """
        1. 从视频中获取全部帧并预处理。
        2. 将倒序帧序列附加在原序列之后，生成双倍长度的帧序列。
        3. 从1帧开始递增，直到使用整个双倍长度的帧序列计算reward，并绘制并保存折线图。
        """
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        frames_tensor = self.padding_video(frames_tensor)
        length = frames_tensor.shape[0]

        # frames_tensor = frames_tensor[:length // 2]

        # 将全部帧的倒序版本拼接到末尾
        reversed_frames_tensor = frames_tensor.flip(dims=[0])  # frames_tensor[::-1] 也可行, 对torch张量使用flip更标准
        
        appended_frames_tensor = th.cat([frames_tensor, reversed_frames_tensor], dim=0)
        #appended_frames_tensor = self.padding_video(appended_frames_tensor)
        save_reversed_video(appended_frames_tensor, output_path="reversed_video.mp4", fps=30)
        new_length = appended_frames_tensor.shape[0]

        incremental_rewards = []

        for i in range(1, new_length + 1):
            sub_frames = appended_frames_tensor[:i]
            with th.no_grad():
                video_embeddings = embedding_image(self.model, self.processor, sub_frames).cuda()
                if self.pca_video_model:
                    video_embeddings = self.pca_video_model.transform(video_embeddings.cpu().numpy())
                    video_embeddings = th.from_numpy(video_embeddings).float().cuda()

            video_embeddings = video_embeddings.view(1, -1, video_embeddings.shape[-1]).float()
            sub_reward = self.transform_model(video_embeddings, None, self.target_embedding).item()
            incremental_rewards.append(sub_reward)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, new_length + 1), incremental_rewards, marker='o')
        plt.title('Reward vs Number of Frames (Original + Reversed)')
        plt.xlabel('Number of Frames Used (Original + Reversed)')
        plt.ylabel('Reward')
        plt.grid(True)

        # 保存图片
        plt.savefig(output_path)
        plt.close()

        gif_buffer = animate_reversed_incremental(appended_frames_tensor, incremental_rewards, fps=15)

        # 如果想要将 gif_buffer 存到本地文件:
        with open(gif_path, 'wb') as f:
            f.write(gif_buffer.getvalue())
        print(f"Saved incremental reversed GIF to {gif_path}")

        return incremental_rewards



state_dict_path = "/scr/jzhang96/roboclip_v2_models_final/RegressionRandom_liv_sample_neg_subtract_after_heads_4_sample_neg_reverse_video_norm_subsample_video/model_149.pth"
state_dict = torch.load(state_dict_path)
saved_args = state_dict.get('args', {})

# 从命令行获取 args
parsed_args = get_args()

# 合并 args，以命令行参数优先
for key, value in vars(parsed_args).items():
    setattr(saved_args, key, value)

args = saved_args
# 打印合并后的 args
print(args)

embedding_dim = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cat_embedding:
    if args.catagorical_progress:
        if args.sample_neg:
            num_bins = args.catagorical_progress_bins + 1
        else:
            num_bins = args.catagorical_progress_bins
        self_attention_model = MultiHeadAttentionConcatenation(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
    else:
        self_attention_model = MultiHeadAttentionConcatenation(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)
else:
    if args.subtract_before:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
            self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
        else:
            self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)
        
    else:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
            self_attention_model = MultiHeadAttentionModel(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
        else:
            self_attention_model = MultiHeadAttentionModel(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)


self_attention_model.load_state_dict(state_dict['model'])
self_attention_model.eval()
device = "cuda" if th.cuda.is_available() else "cpu"
model_name = "liv"
model, processor, tokenizer = load_model(model_name)
model = model.to(device)
model.eval()
pca_video_model = None
target_embedding = embedding_text(model, tokenizer, args.text_string).cuda().float()

video_evaluator = VideoRewardEvaluator(args, model, processor, self_attention_model, target_embedding, pca_video_model)

video_path = "/scr/yusenluo/RoboCLIP/self_collected_vids/window_close/GT/2.mp4"

rewards_dict = video_evaluator.compute_reward(video_path)


print(f"Computed Reward (All Frames): {rewards_dict['all']}")
print(f"Computed Reward (Front Half): {rewards_dict['front_half']}")
print(f"Computed Reward (Back Half): {rewards_dict['back_half']}")
print(f"Computed Reward (Uniform 32): {rewards_dict['uniform_32']}")

incremental_rewards = video_evaluator.test_incremental_frames_reward(video_path, output_path="incremental_reward_plot.png")
print("Incremental Rewards:", incremental_rewards)

incremental_reversed_rewards = video_evaluator.test_incremental_frames_with_reversed(video_path, output_path="incremental_reward_reversed_plot.png")
print("Incremental Rewards with Reversed Frames:", incremental_reversed_rewards)




print("Model loaded")