import torch as th
import joblib
import os
import numpy as np
import cv2
from torchvision.transforms import transforms
from PIL import Image
from self_attention_utils import MultiHeadAttentionSubtraction 
from clip_utils import load_model, embedding_text, embedding_image
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--text_string', type=str, default='opening door')
    parser.add_argument('--pca', action="store_true")
    parser.add_argument('--model_base_path', type=str, default=None)
    parser.add_argument('--transform_model_path', type=str, default="/scr/jzhang96/clip_liv_models/RegressionRandom_liv_subtract_before_heads_4/model_74.pt")
    parser.add_argument('--frame_num', type=int, default=32)
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


class VideoRewardEvaluator:
    def __init__(self, model, processor, transform_model, target_embedding, pca_video_model=None):
        self.model = model
        self.processor = processor
        self.transform_model = transform_model.cuda()
        self.target_embedding = target_embedding.cuda()
        self.pca_video_model = pca_video_model

        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def extract_frames(self, video_path):
        """读取MP4视频文件并提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为RGB格式
            frames.append(frame)
        
        cap.release()
        return frames

    def preprocess_frames(self, frames):
        """裁剪中心区域并转换为Tensor"""
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
        """从视频文件计算reward"""
        # 1. 提取和预处理视频帧
        frames = self.extract_frames(video_path)
        frames_tensor = self.preprocess_frames(frames)
        
        length = frames_tensor.shape[0]
        # frames_tensor = frames_tensor[:length // 2]
        # print(frames_tensor.shape)
        indices = np.linspace(0, length-1, num=9, dtype=int)

        # 对 frames_tensor 进行索引
        frames_tensor = frames_tensor[indices]
        # 2. 计算视频嵌入
        with th.no_grad():
            video_embeddings = embedding_image(self.model, self.processor, frames_tensor).cuda()
            if self.pca_video_model:
                video_embeddings = self.pca_video_model.transform(video_embeddings.cpu().numpy())
                video_embeddings = th.from_numpy(video_embeddings).float().cuda()

        video_embeddings = video_embeddings.view(1, -1, video_embeddings.shape[-1]).float()
        print(video_embeddings.shape)
        # 3. 计算相似度reward
        reward = self.transform_model(video_embeddings, None, self.target_embedding).item()

        return reward


if __name__ == "__main__":
    global args
    args = get_args()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if th.cuda.is_available() else "cpu"
    model_name = "liv"
    model, processor, tokenizer = load_model(model_name)
    model = model.to(device)
    model.eval()
    pca_video_model = None
    if args.pca:
        pca_text_path = os.path.join(args.model_base_path, 'pca_text.pkl') 
        pca_video_path = os.path.join(args.model_base_path, 'pca_video.pkl') 
        pca_text_model = joblib.load(pca_text_path)
        pca_video_model = joblib.load(pca_video_path)

    if args.pca:
        pca_dim = pca_video_model.components_.shape[0]
        transform_model = MultiHeadAttentionSubtraction(pca_dim, num_heads=args.attention_heads)
    else:
        transform_model = MultiHeadAttentionSubtraction(1024, num_heads=args.attention_heads)
    transform_model_path = os.path.join(args.model_base_path, args.transform_model_path)
    dict = th.load(transform_model_path)
    if 'model_state_dict' in dict.keys():
        transform_model.load_state_dict(dict["model_state_dict"])
    else:
        transform_model.load_state_dict(dict)
    transform_model = transform_model.eval().cuda()
    target_embedding = embedding_text(model, tokenizer, args.text_string).cuda().float()
    if args.pca:
        target_embedding = th.from_numpy(pca_text_model.transform(target_embedding.cpu())).cuda().float()
    

    video_evaluator = VideoRewardEvaluator(model, processor, transform_model, target_embedding, pca_video_model)

    # 输入视频路径
    video_path = "/scr/yusenluo/RoboCLIP/self_collected_vids/button_press/GT/step_58880_seed_505_sumreward[39.47]_succ_True.mp4"

    # 计算reward
    reward = video_evaluator.compute_reward(video_path)
    print(f"Computed Reward: {reward}")
