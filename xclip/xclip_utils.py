import os
from transformers import AutoTokenizer, AutoModel, AutoProcessor 
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import torch as th
import pandas as pd
import imageio
import h5py

def readGif(file_path):
    reader = imageio.get_reader(file_path)
    frames = []

    for frame in reader:
        frames.append(np.array(frame))

    return np.stack(frames)


def list_webm_files(folder_path):
    """
    Lists all .webm files within a given folder.

    Parameters:
    - folder_path (str): The path to the folder contains the dataset(webm).

    Returns:
    - list: A list of full paths to the .webm files within the specified folder.
    """
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.webm')])


def load_xclip_model(model_name = "microsoft/xclip-base-patch16-zero-shot", device = "cuda"):
    """
    Default model is "microsoft/xclip-base-patch16-zero-shot" which was trained on Kinetics-400.
    Loads the XCLIP model and processor from Hugging Face.

    Parameters:
    - model_name (str): The name of the model to load.

    Returns:
    - XCLIPTokenizer: The tokenizer for the XCLIP model.
    - XCLIPProcessor: The processor for the XCLIP model.
    - XCLIPModel: The XCLIP model.
    """
    
    # processor = XCLIPProcessor.from_pretrained(model_name)
    # model = XCLIPModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return tokenizer, processor, model

def get_filename_without_extension(file_path):
    """Returns the filename without the extension for a given file path."""
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
    return file_name_without_extension


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

def preprocess_human_demo_xclip(frames):
    """
    Preprocesses frames for video by adjusting size, adding a batch dimension,
    and transposing dimensions to match S3D model input requirements.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Preprocessed frames as a numpy array.
    """
    frames = np.array(frames)
    frames = adjust_size(frames) # (t, 224, 224, 3)
    
    # import pdb; pdb.set_trace()
    # frames = frames[None, :,:,:,:] # (1, t, 224, 224, 3)
    # frames = frames.transpose(0, 4, 1, 2, 3)
    # if frames.shape[1] > 3:
    #     frames = frames[:, :3]
    # frames = frames / 255
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

def adjust_frames_xclip(frames, target_frame_count = 32):
    """
    Ensures same numbers of frames(32). 
    """
    frame_count = frames.shape[0]
    #print(f"frames number{frame_count}")
    frames = th.from_numpy(frames)
    if frame_count < target_frame_count:
        blank_frames = th.zeros(
            (target_frame_count - frame_count, frames.shape[1], frames.shape[2], frames.shape[3]),
            dtype=frames.dtype)
        adjusted_frames = th.cat((frames, blank_frames), dim=0)

    elif frame_count > target_frame_count:
        indices = th.linspace(0, frame_count - 1, target_frame_count, dtype=th.long)
        adjusted_frames = th.index_select(frames, 0, indices)

    else:
        adjusted_frames = frames

    return adjusted_frames


# def adjust_frames_xclip(frames, target_frame_count = 32):
#     """
#     Ensures same numbers of frames(32). 
#     """
#     frame_count = frames.shape[0]
#     #print(f"frames number{frame_count}")
#     if frame_count < target_frame_count:

#         blank_frames = np.zeros(
#             (target_frame_count - frame_count, frames.shape[1], frames.shape[2], frames.shape[3]),
#             dtype=frames.dtype)
#         adjusted_frames = np.concatenate((frames, blank_frames), axis=0)

#     elif frame_count > target_frame_count:
#         indices = np.linspace(0, frame_count - 1, target_frame_count, dtype=np.int)
#         adjusted_frames = frames[indices]

#     else:
#         adjusted_frames = frames

#     return adjusted_frames

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


class TextVideoDataset(Dataset):
    def __init__(self, video_paths, text_label_dict, target_frame_count = 32, model_name = "microsoft/xclip-base-patch16-zero-shot", device = "cuda"):
        # self.video_paths = video_paths
        self.text_label_dict = text_label_dict
        self.video_names = list_webm_files(video_paths)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.target_frame_count = target_frame_count
        self.device = device



    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx] # video absolute path
        video_name = get_filename_without_extension(video_path) # video index
        text = self.text_label_dict[video_name] # text label
        video_frames = read_webm_frames(video_path) # read video frames
        video_frames = preprocess_human_demo_xclip(video_frames)
        video_frames = adjust_frames_xclip(video_frames, self.target_frame_count)
 
        video_frames = [video_frames[i] for i in range (video_frames.shape[0])]
        processor_output = self.processor(videos=video_frames, return_tensors="pt")

        # video_pixel_values = processor_output["pixel_values"].squeeze(0).to(self.device)
        video_pixel_values = processor_output["pixel_values"].squeeze(0)


        return text, video_pixel_values


class OpenXDataset(Dataset):
    def __init__(self, video_folder_path, transform=None, num_samples=None, random_samples=False,
                 csv_path='/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv', dataset_name='droid', debug=False, process_frame = True,
                 model_name="microsoft/xclip-base-patch16-zero-shot"):
        self.transform = transform
        self.random_samples = random_samples
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.video_folder_path = video_folder_path
        self.dataset_name = dataset_name
        self.debug = debug
        self.process_frame = process_frame
        self.processor = AutoProcessor.from_pretrained(model_name)

        if num_samples is not None:
            if random_samples:
                self.df = self.df.sample(n=num_samples)
            else:
                self.df = self.df.head(num_samples)
        if dataset_name == 'fractal':
            self.df = self.df.drop_duplicates(subset=[dataset_name])
        self.df = self.df[self.df[dataset_name].notnull()]

    def __len__(self):
        if self.debug:
            return 30
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row[self.dataset_name].replace(' ', '_')
        #text_label = row['text_label']
        text_label = row[self.dataset_name]
        if self.process_frame:
            video_path = os.path.join(self.video_folder_path, f"{video_id}.gif")
            frames = readGif(video_path)
            frames = frames[:, :, :, :3] # remove alpha channel
            #print(frames.shape) frames: (32, 180, 320, 4)
            
            frames = preprocess_human_demo_xclip(frames)
            frames = adjust_frames_xclip(frames)

            if self.transform:
                frames = self.transform(frames)

            frames = [frames[i] for i in range (frames.shape[0])]
            processor_output = self.processor(videos=frames, return_tensors="pt")

            # video_pixel_values = processor_output["pixel_values"].squeeze(0).to(self.device)
            frames = processor_output["pixel_values"].squeeze(0)
            sample = {'video': frames, 'text': text_label, 'video_id': video_id}

        else:
            sample = {'text': text_label, 'video_id': video_id}
        #print(video_id, text_label)
        return sample



class DroidH5Dataset(Dataset):
    def __init__(self, h5_path, 
                 debug=False, 
                #  process_frame = True,
                 ):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, "r")
        self.debug = debug
        # self.process_frame = process_frame


    def __len__(self):
        if self.debug:
            return 30
        return len(self.h5_file)

    def __getitem__(self, idx):


        group = self.h5_file[str(idx)]
        video_id = group.attrs["video_id"]
        text_label = group.attrs["text"]
        frames = np.asarray(group["frames"][:])
        sample = {'video': frames, 'text': text_label, 'video_id': video_id}

        return sample


class DroidH5LatentDataset(Dataset):
    def __init__(self, h5_path, 
                 debug=False, 
                #  process_frame = True,
                 ):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, "r")
        self.debug = debug
        # self.process_frame = process_frame


    def __len__(self):
        if self.debug:
            return 30
        return len(self.h5_file)

    def __getitem__(self, idx):

        group = self.h5_file[str(idx)]
        text = group.attrs["text"]
        # text_norm_features = np.asarray(group["text_norm_features"][:])
        # video_norm_features = np.asarray(group["video_norm_features"][:])

        text_norm_features = np.asarray(group["text_norm_features"][:])
        video_norm_features = np.asarray(group["video_norm_features"][:])
        sample = {
            'video_latent': video_norm_features, 
            'text_latent': text_norm_features, 
            'text': text,
            }
        return sample