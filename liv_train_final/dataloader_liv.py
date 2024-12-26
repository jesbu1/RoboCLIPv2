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


class LivVideoDataset(Dataset):

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

        if self.args.sample_neg:
            if not self.args.reverse_video:
                if random.random() > 0.75:
                    video_array, progress = self.sample_negative_video_feature(key)
                else:
                    video_array, progress = self.sample_video_feature(key)
            else:
                random_num = random.random()
                if random_num < 0.25:
                    video_array, progress = self.sample_reverse_video_feature(key)
                elif random_num > 0.75:
                    video_array, progress = self.sample_negative_video_feature(key)
                else:
                    video_array, progress = self.sample_video_feature(key)



        else:
            video_array, progress = self.sample_video_feature(key)

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
        text_array = np.asarray(text_dataset[0])
        if self.args.normalize_embedding:
            text_array = np.expand_dims(text_array, axis=0)
            text_array = normalize_embeddings(text_array, return_tensor=False)
            text_array = np.squeeze(text_array, axis=0)
        return text_array
    
    def sample_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        if self.args.normalize_embedding:
            video_frames = normalize_embeddings(video_frames, return_tensor=True)
        else:
            video_frames = th.tensor(video_frames)
        video_length = len(video_frames)
        full_length = len(full_frames)
        progress = video_length / full_length

        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)

        if self.args.catagorical_progress:
            progress = int(progress * self.args.catagorical_progress_bins) 
            if progress == self.args.catagorical_progress_bins:
                progress = self.args.catagorical_progress_bins - 1
            if self.args.sample_neg:
                progress += 1

        return video_frames, progress

    def sample_negative_video_feature(self, env_name):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == env_name:
            negative_env_name = random.choice(self.keys)

        negative_video_group = self.h5_file[self.model_name][negative_env_name]
        negative_datasets = list(negative_video_group.keys())
        negative_random_name = random.choice(negative_datasets)
        negative_video_dataset = np.asarray(negative_video_group[negative_random_name])

        negative_start_idx = random.randint(0, len(negative_video_dataset)-2)
        negative_end_idx = random.randint(negative_start_idx+1, len(negative_video_dataset))

        negative_video_frames = np.array(negative_video_dataset)[negative_start_idx:negative_end_idx]
        if self.args.normalize_embedding:
            negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        progress = 0
        if self.args.subsample_video:
            negative_video_frames = self.padding_video(negative_video_frames, self.args.max_length)


        return negative_video_frames, progress

    def padding_video(self, video_frames, max_length):
        video_length = len(video_frames)
        if type(video_frames) == np.ndarray:
            video_frames = th.tensor(video_frames)
        if video_length < max_length:
            # padding first frame
            padding_length = max_length - video_length
            first_frame = video_frames[0].unsqueeze(0)
            padding_frames = first_frame.repeat(padding_length, 1)
            video_frames = th.cat([padding_frames, video_frames], dim=0)
        
        elif video_length > max_length:
            frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
            video_frames = video_frames[frame_idx]

        return video_frames

    def sample_reverse_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        if self.args.normalize_embedding:
            video_frames = normalize_embeddings(video_frames, return_tensor=True)

        video_length = len(video_frames)
        full_length = len(full_frames)

        if video_length > 1:
            if type(video_frames) == np.ndarray:
                video_frames = th.tensor(video_frames)
            reverse_video_start_idx = random.randint(1, video_length - 1)
            progress = reverse_video_start_idx / full_length # the highest part



            last_video_part = video_frames[reverse_video_start_idx:]
            reverse_last_part = last_video_part.flip(dims=[0])
            # final_part = np.concatenate([video_frames, reverse_last_part], axis=0)
            final_part = th.cat([video_frames, reverse_last_part], dim=0)

            if self.args.subsample_video:
                final_part = self.padding_video(final_part, self.args.max_length)

            if self.args.catagorical_progress:
                progress = int(progress * self.args.catagorical_progress_bins) 
                if progress == self.args.catagorical_progress_bins:
                    progress = self.args.catagorical_progress_bins - 1
                if self.args.sample_neg:
                    progress += 1

            return final_part, progress
        else:
            progress = video_length / full_length
            if self.args.subsample_video:
                video_frames = self.padding_video(video_frames, self.args.max_length)
            
            if self.args.catagorical_progress:
                progress = int(progress * self.args.catagorical_progress_bins) 
                if progress == self.args.catagorical_progress_bins:
                    progress = self.args.catagorical_progress_bins - 1
                if self.args.sample_neg:
                    progress += 1

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
        video = batch[i]["video_array"]
        if type(video) == np.ndarray:
            video = th.tensor(video) 
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
        "progress": th.stack(progress_output),
    }

    return output_dict


