import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from models import MultiHeadAttentionModel, MultiHeadAttentionSubtraction, MultiHeadAttentionConcatenation
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch as th
import io
from clip_utils import load_model, embedding_text, embedding_image
# from confusion_matrix import plot_confusion_matrix_pca, plot_confusion_matrix_pca_class


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--text_string', type=str, default='opening door')
    parser.add_argument('--pca', action="store_true")
    parser.add_argument('--attention_heads', type=int, default=4)

    args = parser.parse_args()
    return args


def animate_reversed_incremental(frames_tensor, incremental_rewards, fps=15):
    """
    Create an animation that shows frames on the left and the incremental reward curve on the right.

    Args:
        frames_tensor: torch.Tensor of shape [N, C, H, W], pixel range [0,1].
        incremental_rewards: list or array of length N, each element is a reward.
        fps: GIF framerate.

    Returns:
        gif_buffer: in-memory BytesIO containing the GIF.
    """
    # 1) Convert frames to numpy for display
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255).astype(np.uint8)
    frames_np = np.transpose(frames_np, (0, 2, 3, 1))

    # 2) Prepare figure
    n = len(frames_np)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    image_plot = ax1.imshow(frames_np[0])
    ax1.set_title('Video Frames')
    ax1.axis('off')

    ax2.set_title('Incremental Rewards')
    ax2.set_xlim(0, n - 1)
    y_min, y_max = min(incremental_rewards), max(incremental_rewards)
    y_range = y_max - y_min if y_max != y_min else 1
    ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    line_plot, = ax2.plot([], [], lw=2, color='blue')
    scat = ax2.scatter([], [], color='red', zorder=5)

    gif_buffer = io.BytesIO()
    images = []

    # 3) Update frames one by one
    for frame_idx in range(n):
        image_plot.set_array(frames_np[frame_idx])
        line_plot.set_data(np.arange(frame_idx + 1), incremental_rewards[:frame_idx + 1])
        scat.set_offsets(np.array([[frame_idx, incremental_rewards[frame_idx]]]))
        fig.canvas.draw()

        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = img_array.reshape((h, w, 3))
        images.append(Image.fromarray(img_array))

    # 4) Save to GIF
    images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
    gif_buffer.seek(0)
    plt.close(fig)
    return gif_buffer


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
        """
        If frames exceed max_length, sample to max_length. Otherwise, pad with the first frame.
        """
        total_frames = frames.shape[0]
        if total_frames > max_length:
            indices = torch.linspace(0, total_frames - 1, max_length).long()
            frames = frames[indices]
        else:
            padding_num = max_length - total_frames
            first_frame = frames[0].unsqueeze(0)
            padding_frames = first_frame.repeat(padding_num, 1, 1, 1)
            frames = torch.cat([padding_frames, frames], dim=0)

        return frames

    def extract_frames(self, video_path):
        """
        Extract frames from video_path using cv2.VideoCapture and return them in RGB format.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def preprocess_frames(self, frames):
        """
        Convert frames (list of np arrays) to torch.Tensor and crop to 224x224 in the center.
        """
        processed_frames = [
            self.transform(Image.fromarray(frame)) for frame in frames
        ]
        print("Processed Frames Shape:", processed_frames[0].shape)
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
        """
        Compute rewards for four different frame subsets of the same video:
        - All frames
        - Front half
        - Back half
        - Uniform 32 frames
        """
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        length = frames_tensor.shape[0]

        frames_all = frames_tensor

        frames_front_half = frames_tensor[: length // 2]
        if self.args.subsample_video:
            frames_front_half = self.padding_video(frames_front_half)

        frames_back_half = frames_tensor[length // 2 :]
        if self.args.subsample_video:
            frames_back_half = self.padding_video(frames_back_half)

        frames_uniform_32 = self.padding_video(frames_tensor)

        print("Frames Tensor Shape (all):", frames_all.shape)
        print("Frames Tensor Shape (front half):", frames_front_half.shape)
        print("Frames Tensor Shape (back half):", frames_back_half.shape)
        print("Frames Tensor Shape (uniform 32):", frames_uniform_32.shape)

        def compute_subreward(sub_frames):
            with th.no_grad():
                video_embeddings = embedding_image(self.model, self.processor, sub_frames).cuda()
                if self.pca_video_model:
                    video_embeddings = self.pca_video_model.transform(video_embeddings.cpu().numpy())
                    video_embeddings = th.from_numpy(video_embeddings).float().cuda()

            video_embeddings = video_embeddings.view(1, -1, video_embeddings.shape[-1]).float()

            if self.args.catagorical_progress:
                return torch.argmax(self.transform_model(video_embeddings, None, self.target_embedding), dim=1).item()
            else:
                return self.transform_model(video_embeddings, None, self.target_embedding).item()

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
        Incrementally add frames from 1 to total, compute reward, and plot the reward curve.
        Then generate a GIF with frames and incremental rewards side by side.
        """
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        if self.args.subsample_video:
            frames_tensor = self.padding_video(frames_tensor)
        length = frames_tensor.shape[0]

        incremental_rewards = []
        for i in range(1, length + 1):
            sub_frames = frames_tensor[:i]
            with th.no_grad():
                video_embeddings = embedding_image(self.model, self.processor, sub_frames).cuda()
                if self.pca_video_model:
                    video_embeddings = self.pca_video_model.transform(video_embeddings.cpu().numpy())
                    video_embeddings = th.from_numpy(video_embeddings).float().cuda()

            video_embeddings = video_embeddings.view(1, -1, video_embeddings.shape[-1]).float()
            if self.args.catagorical_progress:
                sub_reward = torch.argmax(self.transform_model(video_embeddings, None, self.target_embedding), dim=1).item()
            else:
                sub_reward = self.transform_model(video_embeddings, None, self.target_embedding).item()
            incremental_rewards.append(sub_reward)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, length + 1), incremental_rewards, marker='o')
        plt.title('Reward vs Number of Frames')
        plt.xlabel('Number of Frames Used')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

        gif_buffer = animate_reversed_incremental(frames_tensor, incremental_rewards, fps=15)
        with open(gif_path, 'wb') as f:
            f.write(gif_buffer.getvalue())
        print(f"Saved incremental reversed GIF to {gif_path}")

        return incremental_rewards
    
    def test_incremental_frames_with_reversed(self, video_path, output_path="incremental_reward_reversed_plot.png", gif_path="reversed_incremental.gif"):
        """
        Flip the original frames (reverse them), then compute incremental reward from 1 to total reversed frames.
        Finally, generate and save a reward curve and GIF.
        """
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        length = frames_tensor.shape[0]

        # Uncomment or modify if needed: frames_tensor = frames_tensor[:length // 2]

        reversed_frames_tensor = frames_tensor.flip(dims=[0])
        appended_frames_tensor = reversed_frames_tensor
        if self.args.subsample_video:
            appended_frames_tensor = self.padding_video(appended_frames_tensor)

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
            if self.args.catagorical_progress:
                sub_reward = torch.argmax(self.transform_model(video_embeddings, None, self.target_embedding), dim=1).item()
            else:
                sub_reward = self.transform_model(video_embeddings, None, self.target_embedding).item()
            incremental_rewards.append(sub_reward)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, new_length + 1), incremental_rewards, marker='o')
        plt.title('Reward vs Number of Frames (Original + Reversed)')
        plt.xlabel('Number of Frames Used (Original + Reversed)')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

        gif_buffer = animate_reversed_incremental(appended_frames_tensor, incremental_rewards, fps=15)
        with open(gif_path, 'wb') as f:
            f.write(gif_buffer.getvalue())
        print(f"Saved incremental reversed GIF to {gif_path}")

        return incremental_rewards


state_dict_path = "/scr/yusenluo/RoboCLIP/visualization/for_offline_new/roboclip_v2_models_final/RegressionRandom_liv_sample_neg_subtract_after_heads_4_sample_neg_reverse_video_norm/model_149.pth"
state_dict = torch.load(state_dict_path)
saved_args = state_dict.get('args', {})

parsed_args = get_args()

for key, value in vars(parsed_args).items():
    setattr(saved_args, key, value)

args = saved_args
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

video_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos/windowclose/GT/1.gif"

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
