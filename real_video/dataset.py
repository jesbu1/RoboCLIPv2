from torch.utils.data import Dataset, DataLoader
import h5py
from transformers import AutoTokenizer, AutoModel, AutoProcessor
# from s3dg import S3D
import torch as th
import random
import numpy as np
import copy
import json
import pickle
import torch.nn.functional as F

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

def normalize_embeddings(embeddings, return_tensor=False):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()



class RealVideoDataset(Dataset):

    def __init__(self, args, h5_file, train=True):
        self.h5_file = h5_file
        # eval_idx = args.eval_task
        # train_idx = [i for i in range(8) if i not in eval_idx]
        train_idx = [i for i in range(8)]
        self.args = args

        total_key_name = list(self.h5_file["text_embeddings"].keys())
        train_keys = [total_key_name[i] for i in train_idx]
        # eval_keys = [total_key_name[i] for i in eval_idx]
        # import pdb; pdb.set_trace()
        if train:
            self.keys = train_keys
        # else:
        #     self.keys = eval_keys

        self.args = args


    def __len__(self):
        return len(self.keys) * 1500


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        text_array = np.asarray(self.h5_file["text_embeddings"][key])
        

        if self.args.sample_negative:
            if random.random() > self.args.negative_weight:
                video_array, progress = self.sample_progress_video_feature(key)
            else:
                video_array, progress = self.sample_negative_video_feature(key)
        else:
            video_array, progress = self.sample_progress_video_feature(key)
        
        text_array = text_array.squeeze(0)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress
        }

        return  output_dict

    def sample_progress_video_feature(self, key):

        video_group = self.h5_file["video_embeddings"][key]
        video_file_name = list(video_group.keys())
        video_file_name = random.choice(video_file_name)
        video_array = np.asarray(video_group[video_file_name])
        video_array = video_array.squeeze(1)
        # reverse the video
        video_array = video_array[::-1]
        # start_range = int(len(video_array) // 5)
        start_range = 0
        start_idx = 0
        # start_idx = random.randint(start_range, len(video_array)-2)
        end_idx = random.randint(start_idx+1, len(video_array))

        video_frames = np.array(video_array)[start_idx:end_idx]
        length = len(video_array) - start_idx
        progress = video_frames.shape[0] / length
        return video_frames, progress


    def sample_negative_video_feature(self, key):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == key:
            negative_env_name = random.choice(self.keys)
        video_group = self.h5_file["video_embeddings"][key]
        video_file_name = list(video_group.keys())
        video_file_name = random.choice(video_file_name)
        video_array = np.asarray(video_group[video_file_name])
        video_array = video_array.squeeze(1)
        # reverse the video
        video_array = video_array[::-1]
        start_idx = random.randint(0, len(video_array)-2)
        end_idx = random.randint(start_idx+1, len(video_array))

        video_frames = np.array(video_array)[start_idx:end_idx]

        progress = 0
        return video_frames, progress



    def __del__(self):
        self.h5_file.close()




def video_collate_fn(batch):
    # Find the maximum video length (number of frames) in the batch

    length = [data["video_array"].shape[0] for data in batch]
    max_length = max(length)

    embedding_size = batch[0]["video_array"].shape[1]
    batch_size = len(batch)

    
    video_output = list()
    mask_output = list()
    text_output = list()
    progress_output = list()
    for i in range(batch_size):
        # print(i, batch[i]["video_array"].shape)
        video = batch[i]["video_array"]
        # video = batch[i]["video_array"]
        padding = np.zeros((max_length - video.shape[0], embedding_size))
        
        padded_video = np.concatenate([video, padding], axis=0)
        padded_video = th.tensor(padded_video)
        mask = th.zeros(max_length)
        mask[:video.shape[0]] = 1
        text = th.tensor(batch[i]["text_array"])
        progress = th.tensor(batch[i]["progress"])
        

        video_output.append(padded_video)
        mask_output.append(mask)
        text_output.append(text)
        progress_output.append(progress)
    output_dict = {
        "video_array": th.stack(video_output),
        "mask": th.stack(mask_output),
        "text_array": th.stack(text_output),
        "progress": th.stack(progress_output)
    }

    return output_dict



class LastFrameDataset(Dataset):

    def __init__(self, args, h5_file, train=True, split = False):
        self.h5_file = h5_file
        # eval_idx = args.eval_task
        # train_idx = [i for i in range(8) if i not in eval_idx]
        train_idx = [i for i in range(8)]
        self.args = args

        total_key_name = list(self.h5_file["text_embeddings"].keys())
        train_keys = [total_key_name[i] for i in train_idx]

        self.keys = train_keys
        # else:
        #     self.keys = eval_keys

        self.args = args
        self.train = train
        self.split = split


    def __len__(self):
        if self.train:
            return len(self.keys) * 100
        else:
            return len(self.keys) * 5
        


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        text_array = np.asarray(self.h5_file["text_embeddings"][key])
        

        if random.random() > 0.5:
            video_array = self.sample_progress_video_feature(key)
            label = 1
        else:
            video_array = self.sample_negative_video_feature(key)
            label = 0

        
        text_array = text_array.squeeze(0)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "label": label,
        }

        return  output_dict

    def sample_progress_video_feature(self, key):

        video_group = self.h5_file["video_embeddings"][key]
        video_file_name = list(video_group.keys())
        if self.split:
            if self.train:
                video_file_name = video_file_name[:15]
            else:
                video_file_name = video_file_name[15:]
        video_file_name = random.choice(video_file_name)
        video_array = np.asarray(video_group[video_file_name])
        video_array = video_array.squeeze(1)
        video_frames = np.array(video_array)[-1]

        return video_frames


    def sample_negative_video_feature(self, key):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == key:
            negative_env_name = random.choice(self.keys)
        video_group = self.h5_file["video_embeddings"][key]
        video_file_name = list(video_group.keys())
        if self.split:
            if self.train:
                video_file_name = video_file_name[:15]
            else:
                video_file_name = video_file_name[15:]
        video_file_name = random.choice(video_file_name)
        video_array = np.asarray(video_group[video_file_name])
        video_array = video_array.squeeze(1)
        # video_frames = np.array(video_array)[-1]
        video_frames = np.array(video_array)[0]

        return video_frames



class LastFrameImageDataset(Dataset):

    def __init__(self, args, train=True, split = False):

        video_base_dir = '/scr/jzhang96/jesse_datasets_videos'


        self.h5_file = h5_file
        train_idx = [i for i in range(8)]
        self.args = args

        total_key_name = list(self.h5_file["text_embeddings"].keys())
        train_keys = [total_key_name[i] for i in train_idx]

        self.keys = train_keys


        self.args = args
        self.train = train
        self.split = split


    def __len__(self):
        if self.train:
            return len(self.keys) * 100
        else:
            return len(self.keys) * 5
        


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        text_array = np.asarray(self.h5_file["text_embeddings"][key])
        

        if random.random() > 0.5:
            video_array = self.sample_progress_video_feature(key)
            label = 1
        else:
            video_array = self.sample_negative_video_feature(key)
            label = 0

        
        text_array = text_array.squeeze(0)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "label": label,
        }

        return  output_dict

    def sample_progress_video_feature(self, key):

        video_group = self.h5_file["video_embeddings"][key]
        video_file_name = list(video_group.keys())
        if self.split:
            if self.train:
                video_file_name = video_file_name[:15]
            else:
                video_file_name = video_file_name[15:]
        video_file_name = random.choice(video_file_name)
        video_array = np.asarray(video_group[video_file_name])
        video_array = video_array.squeeze(1)
        video_frames = np.array(video_array)[-1]

        return video_frames


    def sample_negative_video_feature(self, key):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == key:
            negative_env_name = random.choice(self.keys)
        video_group = self.h5_file["video_embeddings"][key]
        video_file_name = list(video_group.keys())
        if self.split:
            if self.train:
                video_file_name = video_file_name[:15]
            else:
                video_file_name = video_file_name[15:]
        video_file_name = random.choice(video_file_name)
        video_array = np.asarray(video_group[video_file_name])
        video_array = video_array.squeeze(1)
        video_frames = np.array(video_array)[-1]

        return video_frames
