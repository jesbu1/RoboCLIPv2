import PIL
import imageio
from torch.utils.data import Dataset, DataLoader
import torch as th
import json
import numpy as np
import os
import cv2
import random
import pandas as pd



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


class OpenXDataset(Dataset):
    def __init__(self, video_folder_path, transform=None, num_samples=None, random_samples=False,
                 csv_path='/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv', dataset_name='droid'):
        self.transform = transform
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
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row[self.dataset_name].replace(' ', '_')
        #text_label = row['text_label']
        text_label = row[self.dataset_name]
        video_path = os.path.join(self.video_folder_path, f"{video_id}.gif")
        frames = readGif(video_path)
        #print(frames.shape)
        frames = preprocess_human_demo(frames)
        frames = adjust_frames(frames)

        if self.transform:
            frames = self.transform(frames)

        sample = {'video': frames, 'text': text_label, 'video_id': video_id}
        #print(video_id, text_label)
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