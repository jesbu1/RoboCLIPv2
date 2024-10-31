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


class ClipLivVideoMeanDataset(Dataset):

    def __init__(self, args, h5_file):
        self.h5_file = h5_file
        subset_list = json.load(open("task_subset.json"))
        subset_name = "subset_6"
        self.keys = subset_list[subset_name]
        self.model_name = args.model_name
        self.args = args


    def __len__(self):
        return len(self.keys) * 1000


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)
        video_array, progress = self.sample_progress_video_feature(key)


        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress
        }



        return  output_dict
    
    def sample_text_feature(self, env_name):
        text_env_name = env_name + "_text"
        text_dataset = self.h5_file[self.model_name][text_env_name]
        # choose index
        idx = random.randint(0, len(text_dataset)-1)
        text_array = np.asarray(text_dataset[idx])
        return text_array

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset)-1)

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=False)
        length = len(progress_dataset)  - start_idx
        progress = (end_idx - start_idx + 1) / length
        video_frames = np.mean(video_frames, axis=0)

        return video_frames, progress

    def __del__(self):
        self.h5_file.close()



class ClipLivVideoDataset(ClipLivVideoMeanDataset):

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset)-1)

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(progress_dataset) - start_idx
        progress = (end_idx - start_idx + 1) / length

        return video_frames, progress


class ClipLivVideoReverseDataset(ClipLivVideoDataset):

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset)-1)

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(progress_dataset) - start_idx
        progress = (end_idx - start_idx + 1) / length
        reverse = random.random() > 0.5
        # if reverse:
        if reverse:
            # video_frames = th.flip(video_frames, [0])
            frames = []
            for i in range(len(video_frames) - 1, -1, -1):
                frames.append(video_frames[i])
            video_frames = th.stack(frames)
            progress = -progress

        return video_frames, progress



class ClipLivVideoCatDataset(ClipLivVideoMeanDataset):

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset)-1)

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=False)

        if len(video_frames) < self.args.sample_frames:
            # padding 1st frame
            num_pads = self.args.sample_frames - len(video_frames)
            first_frame = video_frames[0:1]
            video_frames = np.concatenate([first_frame]*num_pads + [video_frames], axis=0)

        elif len(video_frames) > self.args.sample_frames:
            # sample frames
            first_frame = video_frames[0:1]
            last_frame = video_frames[-1:]
            mid_frames = video_frames[1:-1]
            num_samples = self.args.sample_frames - 2
            frame_idx = np.linspace(0, len(mid_frames) - 1, num_samples, dtype=int)
            # even sample from frame_idx
            choose_idx = random.sample(list(frame_idx), num_samples)
            choose_idx = np.sort(choose_idx)
            mid_frames = mid_frames[choose_idx]
            video_frames = np.concatenate([first_frame, mid_frames, last_frame], axis=0)

        length = len(progress_dataset) - start_idx
        progress = (end_idx - start_idx + 1) / length

        return video_frames, progress




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
        padding = th.zeros((max_length - video.shape[0], embedding_size))
        padded_video = th.cat((video, padding), dim=0)
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


