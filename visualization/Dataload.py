import PIL
import imageio
from torch.utils.data import Dataset, DataLoader
import torch as th
import json
import numpy as np
import os
import cv2
import re
import random
import pandas as pd


device = th.device("cuda" if th.cuda.is_available() else "cpu")

class VideoTextDataset(Dataset):
    def __init__(self, video_paths, transform=None, num_samples=None, random_samples=False, dataset_type = 'train', dataset = 'Sth'):
        self.video_paths = video_paths
        self.transform = transform
        self.random_samples = random_samples
        self.dataset_type = dataset_type
        self.dataset = dataset
        if num_samples is not None:
            if random_samples:
                self.indices = random.sample(range(len(self.video_paths)), num_samples)
            else:
                self.indices = list(range(num_samples))
        else:
            self.indices = list(range(len(self.video_paths)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        video_idx = self.indices[idx]
        video_path = self.video_paths[video_idx]
        video_id = get_filename_without_extension(video_path)
        if self.dataset == 'OpenX':
            text_label = video_id.replace('_',' ')
            print(text_label)
            frames = readGif(video_path)
        else:
            text_label = find_label(video_id, self.dataset_type)
            frames = read_webm_frames(video_path)
        frames = preprocess_human_demo(frames)
        frames = adjust_frames(frames)
        print(frames.shape)
        if self.transform:
            frames = self.transform(frames)

        sample = {'video': frames, 'text': text_label, 'video_id': video_id}

        return sample


class SthDataset(Dataset):
    def __init__(self, video_folder_path, transform=None, num_samples=None, random_samples=False,
                 csv_path='/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv'):
        self.transform = transform
        self.random_samples = random_samples
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.video_folder_path = video_folder_path

        if num_samples is not None:
            if random_samples:
                self.df = self.df.sample(n=num_samples)
            else:
                self.df = self.df.head(num_samples)
        self.df = self.df[self.df['OpenX'].notnull()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(int(row['video_id']))
        text_label = row['OpenX']

        video_path = os.path.join(self.video_folder_path, f"{video_id}.webm")
        frames = read_webm_frames(video_path)

        frames = preprocess_human_demo(frames)
        frames = adjust_frames(frames)

        if self.transform:
            frames = self.transform(frames)

        sample = {'video': frames, 'text': text_label, 'video_id': video_id}
        #print(video_id, text_label)
        return sample


import kornia.augmentation as K
import torch.nn as nn


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


aug_cfg = AttrDict(
    brightness=0.3,
    contrast=0.3,
    saturation=0.3,
    hue=0.3,
    color_p=0.9,
    noise_std=0.1,
    noise_p=0.0,
    channel_shuffle_p=0.0,
    degrees=15,
    translate=0.1,
    affine_p=0.6,
    erase_p=0.1,
)
aug_cfg_strong = AttrDict(
    brightness = 0.5,
    contrast = 0.5,
    saturation = 0.5,
    hue = 0.5,
    color_p = 0.9,
    noise_std = 0.2,
    noise_p = 0.2,
    channel_shuffle_p = 0.2,
    degrees = 25,
    translate = 0.2,
    affine_p = 0.8,
    erase_p = 0.25,
)
aug_cfg_weak = AttrDict(
    brightness=0.1,
    contrast=0.1,
    saturation=0.1,
    hue=0.1,
    color_p=0.5,
    noise_std=0.05,
    noise_p=0.1,
    channel_shuffle_p=0.0,
    degrees=5,
    translate=0.05,
    affine_p=0.3,
    erase_p=0.05,
)

class AugmentationPipeline(nn.Module):
    def __init__(self, device, strength='normal'):
        super(AugmentationPipeline, self).__init__()
        self.device = device
        if strength == 'weak':
            cfg = aug_cfg_weak
        elif strength == 'strong':
            cfg = aug_cfg_strong
        else:
            cfg = aug_cfg
        self.augmentations = nn.Sequential(
            K.ColorJitter(
                cfg.brightness,
                cfg.contrast,
                cfg.saturation,
                cfg.hue,
                p=cfg.color_p,
            ),
            K.RandomGaussianNoise(std=cfg.noise_std, p=cfg.noise_p),
            K.RandomChannelShuffle(p=cfg.channel_shuffle_p),
            K.RandomAffine(
                degrees=cfg.degrees,
                translate=(cfg.translate, cfg.translate),
                p=cfg.affine_p,
            ),
            K.RandomErasing(p=cfg.erase_p),
        ).to(self.device)

    def forward(self, x):
        # 将视频张量从 (1, 3, 32, 224, 224) 转换为 (32, 3, 224, 224)
        x = x.squeeze(0).permute(1, 0, 2, 3)  # (32, 3, 224, 224)
        #x = x.to(self.device)
        # 使用视频的第一帧生成增强参数
        first_frame = x[0].unsqueeze(0)  # (1, 3, 224, 224)
        video_params = {}
        for name, aug in self.augmentations.named_children():
            # 为每个增强操作设置不同的随机种子
            seed = random.randint(0, 100000)
            th.manual_seed(seed)
            random.seed(seed)

            # 获取增强参数
            params = aug.forward_parameters(first_frame.shape)
            # 根据参数的性质,决定是否复制32次
            video_params[name] = {}
            for k, v in params.items():
                if k in ['order']:
                    # 这些参数在所有帧中保持不变
                    video_params[name][k] = v
                else:
                    # 其他参数复制32次,以匹配视频帧数
                    video_params[name][k] = v.repeat(x.shape[0], *[1] * (v.dim() - 1))

            # 修改 forward_input_shape 以匹配整个视频张量的形状
            video_params[name]['forward_input_shape'] = th.tensor(x.shape)
        #print(video_params)
            # 对整个视频张量应用相同的增强参数
        for name, params in video_params.items():
            x = self.augmentations[int(name)](x, params=params)

            # 转换回 (1, 3, 32, 224, 224)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 32, 224, 224)
        return x.cpu()
    #batch (10)处理：
    # def forward(self, x):
    #     # 将视频张量从 (10, 3, 32, 224, 224) 转换为 (320, 3, 224, 224)
    #     batch_size = x.shape[0]
    #     print(x.shape) #torch.Size([10, 1, 3, 32, 224, 224])
    #     x = x.squeeze(1).permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)  # (320, 3, 224, 224)
    #     print(x.shape)
    #     x = x.to(self.device)
    # 
    #     # 使用视频的第一帧生成增强参数
    #     #first_frames = x.view(batch_size, -1, *x.shape[1:])[:, 0, ...] # (10, 3, 224, 224)
    #     #print(first_frames.shape)
    #     video_params = {}
    # 
    #     for name, aug in self.augmentations.named_children():
    #         # 为每个增强操作设置不同的随机种子
    #         seed = random.randint(0, 100000)
    #         th.manual_seed(seed)
    #         random.seed(seed)
    # 
    #         # 获取增强参数
    #         params = aug.forward_parameters(x.permute(0, 2, 1, 3, 4)[:, 0, ...])
    #         
    #         # 根据参数的性质,决定是否复制32次
    #         video_params[name] = {}
    #         for k, v in params.items():
    #             video_params[name][k] = v
    #             # if k in ['order']:
    #             #     # 这些参数在所有帧中保持不变
    #             #     video_params[name][k] = v
    #             # else:
    #             #     # 其他参数复制32次,以匹配视频帧数
    #             #     video_params[name][k] = v.repeat_interleave(32, dim=0)
    # 
    #         # 修改 forward_input_shape 以匹配整个视频张量的形状
    #         #video_params[name]['forward_input_shape'] = th.tensor((320, 3, 224, 224))
    #     print(video_params)
    # 
    #     # 检查参数列表
    #     for name, params in video_params.items():
    #         for k, v in params.items():
    #             if k not in ['order', 'forward_input_shape']:
    #                 assert v.shape[0] == 10, f"{name}, {k}: {v.shape}"
    #                 for i in range(10):
    #                     assert (v[i * 32:(i + 1) * 32] == v[
    #                         i * 32]).all(), f"{name}, {k}, group {i}: {v[i * 32:(i + 1) * 32]}"
    # 
    #     # 对整个视频张量应用相同的增强参数
    #     for name, params in video_params.items():
    #         x = self.augmentations[int(name)](x, params=params)
    # 
    #     # 将增强后的视频张量转换回 (10, 3, 32, 224, 224)
    #     augmented_videos = x.view(batch_size, 32, *x.shape[1:]).permute(0, 2, 1, 3, 4)
    #     print('after augmentation', augmented_videos.shape)
    #     return augmented_videos.cpu()
    #按照frames 处理
    # def forward(self, x):
    #     # 将视频张量从 (1, 3, 32, 224, 224) 转换为 (32, 3, 224, 224)
    #     x = x.squeeze(0).permute(1, 0, 2, 3)  # (32, 3, 224, 224)
    # 
    #     # 使用视频的第一帧生成增强参数
    #     first_frame = x[0].unsqueeze(0)  # (1, 3, 224, 224)
    #     video_params = {}
    #     for name, aug in self.augmentations.named_children():
    #         # 为每个增强操作设置不同的随机种子
    #         seed = random.randint(0, 100000)
    #         th.manual_seed(seed)
    #         random.seed(seed)
    # 
    #         # 获取增强参数
    #         params = aug.forward_parameters(first_frame.shape)
    # 
    #         video_params[name] = params
    #     #print(video_params)
    # 
    #     # 对所有帧应用相同的增强参数
    #     augmented_frames = []
    #     for frame in x:
    #         frame = frame.unsqueeze(0)  # (1, 3, 224, 224)
    #         for name, params in video_params.items():
    #             frame = self.augmentations[int(name)](frame, params=params)
    #         augmented_frames.append(frame.squeeze(0))
    # 
    #     augmented_video = th.stack(augmented_frames, dim=0)  # (32, 3, 224, 224)
    # 
    #     # 转换回 (1, 3, 32, 224, 224)
    #     augmented_video = augmented_video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 32, 224, 224)
    #     return augmented_video



class OpenXDataset(Dataset):
    def __init__(self, video_folder_path, transform=None, num_samples=None, random_samples=False,
                 csv_path='/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv', dataset_name='droid', num_augmented_samples=0):
        self.transform = transform
        self.num_augmented_samples = num_augmented_samples
        self.random_samples = random_samples
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.video_folder_path = video_folder_path
        self.dataset_name = dataset_name

        if num_samples is not None:
            if random_samples:
                self.df = self.df.sample(n=num_samples)
            else:
                self.df = self.df.head(num_samples)
        if dataset_name == 'fractal':
            self.df = self.df.drop_duplicates(subset=[dataset_name])
        self.df = self.df[self.df[dataset_name].notnull()]

    def __len__(self):
        return len(self.df) * (1 + self.num_augmented_samples)  # 原始样本加上增强样本

    def __getitem__(self, idx):
        original_idx = idx // (1 + self.num_augmented_samples)
        aug_idx = idx % (1 + self.num_augmented_samples)

        row = self.df.iloc[original_idx]
        video_id = row[self.dataset_name].replace(' ', '_')
        text_label = row[self.dataset_name]
        video_path = os.path.join(self.video_folder_path, f"{video_id}.gif")
        frames = readGif(video_path)
        frames = preprocess_human_demo(frames)
        frames = adjust_frames(frames)

        if aug_idx > 0 and self.transform:  # 对于增强版本并且有transform时
            frames = self.transform(frames)

        sample = {'video': frames, 'text': text_label, 'video_id': video_id}
        return sample



class EmbeddingsDataset(Dataset):
    def __init__(self, video_embeddings, text_embeddings, video_ids, text_labels):
        self.video_embeddings = video_embeddings
        self.text_embeddings = text_embeddings
        self.video_ids = video_ids
        self.text_labels = text_labels

    def __len__(self):
        return len(self.video_embeddings)

    def __getitem__(self, idx):
        video_embedding = self.video_embeddings[idx]
        text_embedding = self.text_embeddings[idx]
        video_id = self.video_ids[idx]
        text_label = self.text_labels[idx]
        return {"video_embedding": video_embedding,
                "text_embedding": text_embedding,
                "video_id": video_id,
                "text_label": text_label}


def find_label(video_filename, dataset_name):
    """
    Finds the corresponding label for a given video filename based on the dataset name.

    Parameters:
    - video_filename: Name of the video file (without the extension).
    - dataset_name: Name of the dataset ('train', 'validate', 'test').

    Returns:
    - The found label as a string, or None if not found. If the dataset is 'test', always returns None since labels are not available.
    """
    json_path_map = {
        'train': '../labels/train.json',
        'validation': '../labels/validation.json',
        'test': None  # Assuming test.json does not contain labels
    }

    json_path = json_path_map.get(dataset_name)

    if json_path is None:
        print(f"No label file for dataset '{dataset_name}'.")
        return None

    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            for item in data:
                if item['id'] == video_filename:
                    return item.get('label')  # Using .get in case 'label' key is missing

    except FileNotFoundError:
        print(f"File '{json_path}' not found.")
    except json.JSONDecodeError:
        print(f"File '{json_path}' is not valid JSON.")

    return None

def get_filename_without_extension(file_path):
    """Returns the filename without the extension for a given file path."""
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
    return file_name_without_extension

def preprocess_human_demo(frames):
    """
    Preprocesses frames for video by adjusting size, adding a batch dimension,
    and transposing dimensions to match S3D model input requirements.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Preprocessed frames as a numpy array.
    """
    frames = np.array(frames)
    frames = adjust_size(frames)
    frames = frames[None, :,:,:,:]
    frames = frames.transpose(0, 4, 1, 2, 3)
    if frames.shape[1] > 3:
        frames = frames[:, :3]
    frames = frames / 255
    return frames

def adjust_frames(frames, target_frame_count = 32):
    """
    Ensures same numbers of frames(32).
    """
    _, _, frame_count, _, _ = frames.shape
    #print(f"frames number{frame_count}")
    frames = th.from_numpy(frames)
    if frame_count < target_frame_count:
        blank_frames = th.zeros(
            (frames.shape[0], frames.shape[1], target_frame_count - frame_count, frames.shape[3], frames.shape[4]),
            dtype=frames.dtype)
        adjusted_frames = th.cat((frames, blank_frames), dim=2)

    elif frame_count > target_frame_count:
        indices = th.linspace(0, frame_count - 1, target_frame_count, dtype=th.long)
        adjusted_frames = th.index_select(frames, 2, indices)
    else:
        adjusted_frames = frames

    return adjusted_frames

def adjust_size(frames):
    """
    Adjusts the size of the frames to a target height and width by cropping.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Cropped frames as a numpy array.
    """
    if len(frames) == 0:
        return np.array([])

    target_height = 224
    target_width = 224

    height, width, _ = frames[0].shape

    if height < target_height or width < target_width:
        adjusted_frames = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR) for frame in
                           frames]
    else:
        start_x = width // 2 - target_width // 2
        start_y = height // 2 - target_height // 2
        adjusted_frames = [frame[start_y:start_y + target_height, start_x:start_x + target_width] for frame in frames]

    return np.array(adjusted_frames)

def read_webm_frames(video_path):
    """
    Reads frames from a .webm video file using OpenCV.

    Parameters:
    - video_path: Path to the video file.

    Returns:
    - frames: A list of video frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert color space from BGR to RGB since OpenCV uses BGR by default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames

def list_webm_files(folder_path):
    """
    Lists all .webm files within a given folder.

    Parameters:
    - folder_path (str): The path to the folder contains the dataset(webm).

    Returns:
    - list: A list of full paths to the .webm files within the specified folder.
    """
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.webm')])


class TextEmbedding:
    def __init__(self, 
            token_to_word_path="./s3d_dict.npy",
            max_words=16,
            ):
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent.lower())) for sent in x]
        return th.stack(split_x, dim=0)


def Embedding(model, data_loader):
    """
    Generates embeddings for video and text using a given model and accumulates them along with video IDs and text labels into a dataset.

    Parameters:
    - model: The model used for generating embeddings.
    - data_loader: DataLoader providing batches of videos and corresponding text labels.

    Returns:
    - embeddings_dataset: A custom dataset containing all video embeddings, text embeddings, video IDs, and text labels.
    - mappings: A dictionary containing mappings from video IDs to text labels, and indices to video IDs and text labels.
    """
    all_video_embeddings = []
    all_text_embeddings = []
    video_ids, text_labels = [], []
    for batch in data_loader:
        videos = th.squeeze(batch['video'].float(), dim=1)
        text_labels_batch = batch['text']
        video_ids_batch = batch['video_id']

        with th.no_grad():
            video_output = model(videos)
            text_output = model.text_module(text_labels_batch)

            all_video_embeddings.append(video_output['video_embedding'])
            all_text_embeddings.append(text_output['text_embedding'])

        video_ids.extend(video_ids_batch)
        text_labels.extend(text_labels_batch)

    all_video_embeddings = th.cat(all_video_embeddings, dim=0)
    all_text_embeddings = th.cat(all_text_embeddings, dim=0)

    # Construct mappings
    mappings = {
        "video_id_to_text_label": dict(zip(video_ids, text_labels)),
        "index_to_video_id": {i: vid for i, vid in enumerate(video_ids)},
        "index_to_text_label": {i: lbl for i, lbl in enumerate(text_labels)}
    }

    embeddings_dataset = EmbeddingsDataset(all_video_embeddings.cpu(), all_text_embeddings.cpu(), video_ids,
                                           text_labels)
    return all_video_embeddings, all_text_embeddings, embeddings_dataset, mappings



def Embedding_gpu(model, data_loader):
    """
    Generates embeddings for video and text using a given model and accumulates them along with video IDs and text labels into a dataset.

    Parameters:
    - model: The model used for generating embeddings.
    - data_loader: DataLoader providing batches of videos and corresponding text labels.

    Returns:
    - embeddings_dataset: A custom dataset containing all video embeddings, text embeddings, video IDs, and text labels.
    - mappings: A dictionary containing mappings from video IDs to text labels, and indices to video IDs and text labels.
    """
    all_video_embeddings = []
    all_text_embeddings = []
    video_ids, text_labels = [], []
    text_embedding = TextEmbedding(token_to_word_path="../s3d_dict.npy")
    for batch in data_loader:
        videos = th.squeeze(batch['video'].float(), dim=1)
        print(videos.shape)
        text_labels_batch = batch['text']
        video_ids_batch = batch['video_id']
        text_emb = text_embedding._words_to_ids(text_labels_batch)

        videos = videos.to(device)  # 将数据移动到 GPU 上
        text_emb = text_emb.to(device)

        with th.no_grad():
            video_output = model(videos)
            text_output = model.text_module(text_emb)

            all_video_embeddings.append(video_output['video_embedding'])
            all_text_embeddings.append(text_output['text_embedding'])

        video_ids.extend(video_ids_batch)
        text_labels.extend(text_labels_batch)

    all_video_embeddings = th.cat(all_video_embeddings, dim=0)
    all_text_embeddings = th.cat(all_text_embeddings, dim=0)

    # Construct mappings
    mappings = {
        "video_id_to_text_label": dict(zip(video_ids, text_labels)),
        "index_to_video_id": {i: vid for i, vid in enumerate(video_ids)},
        "index_to_text_label": {i: lbl for i, lbl in enumerate(text_labels)}
    }

    embeddings_dataset = EmbeddingsDataset(all_video_embeddings.cpu(), all_text_embeddings.cpu(), video_ids,
                                           text_labels)
    return all_video_embeddings, all_text_embeddings, embeddings_dataset, mappings


def filter_top_embeddings(video_embeddings, text_embeddings, top_portion):

    video_embeddings_norm = video_embeddings / video_embeddings.norm(dim=1, keepdim=True)
    text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    similarities = (video_embeddings_norm * text_embeddings_norm).sum(dim=1)

    top_k = int(len(video_embeddings) * top_portion)
    _, top_indices = th.topk(similarities, top_k, largest=False)

    top_video_embeddings = video_embeddings[top_indices]
    top_text_embeddings = text_embeddings[top_indices]
    print(top_video_embeddings.shape)

    return top_video_embeddings, top_text_embeddings


def readGif(file_path):
    reader = imageio.get_reader(file_path)
    frames = []

    for frame in reader:
        frames.append(np.array(frame))
    return np.stack(frames)

def readAvi(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


