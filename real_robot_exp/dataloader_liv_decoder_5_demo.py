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
        embeddings = th.tensor(embeddings).float()
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


class LivVideoDecoderDataset5Frames(Dataset):

    def __init__(self, args, h5_file):
        self.h5_file = h5_file["train"]
        self.args = args
        self.keys = list(self.h5_file.keys())
        for key in self.keys:
            if key.endswith("text"):
                self.keys.remove(key)

    def __len__(self):
        return self.args.batch_size * 100
        # return len(self.keys) * 5
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)

        # if self.args.sample_neg:
        #     if not self.args.reverse_video:
        #         if random.random() > 0.75:
        #             video_array, progress, class_label = self.sample_negative_video_feature(key)
        #         else:
        #             video_array, progress, class_label = self.sample_video_feature(key)
        #     elif not self.args.fully_reverse_data:
        random_num = random.random()
        if random_num < 0.25:
            video_array, progress, class_label = self.sample_reverse_video_feature(key)
        elif random_num > 0.50:
            video_array, progress, class_label = self.sample_negative_video_feature(key)
        else:
            video_array, progress, class_label = self.sample_video_feature(key)
        #     else:
        #         random_num = random.random()
        #         if random_num < 0.20:
        #             video_array, progress, class_label = self.sample_reverse_video_feature(key)
        #         elif random_num < 0.30:
        #             video_array, progress, class_label = self.sample_fully_reverse_video_feature(key)
        #         elif random_num < 0.50:
        #             video_array, progress, class_label = self.sample_negative_video_feature(key)
        #         else:
        #             video_array, progress, class_label = self.sample_video_feature(key)



        # else:
        #     video_array, progress, class_label = self.sample_video_feature(key)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "class_label": class_label
        }
        return  output_dict

    def sample_text_feature(self, env_name):

        text_env_name = env_name + "_text"

        text_dataset = self.h5_file[text_env_name]
        # choose index
        idx = random.randint(0, len(text_dataset)-1)
        text_array = np.asarray(text_dataset[idx])
        if self.args.normalize_embedding:
            text_array = np.expand_dims(text_array, axis=0)
            text_array = normalize_embeddings(text_array, return_tensor=False)
            text_array = np.squeeze(text_array, axis=0)
        return text_array
    
    def sample_video_feature(self, env_name):
        progress_group = self.h5_file[env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        # random_name = "0"
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-7)
        end_idx = random.randint(start_idx+5, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        if self.args.normalize_embedding:
            video_frames = normalize_embeddings(video_frames, return_tensor=True)
        else:
            video_frames = th.tensor(video_frames)
        # video_length = len(video_frames)
        full_length = len(full_frames)
        video_progress = np.arange(0, video_frames.shape[0]) + 1
        video_progress = video_progress / full_length
        # progress = video_length / full_length

        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)
            video_progress = np.expand_dims(video_progress, axis=1)
            video_progress = self.padding_video(video_progress, self.args.max_length).detach().cpu().numpy()
            video_progress = np.squeeze(video_progress, axis=1)

        if self.args.catagorical_progress:

            video_progress = np.floor(video_progress * self.args.catagorical_progress_bins) 
            if video_progress[-1] == self.args.catagorical_progress_bins:
                video_progress[-1] = self.args.catagorical_progress_bins - 1
            if self.args.sample_neg:
                video_progress += 1

        return video_frames, video_progress, np.ones(video_frames.shape[0])

    def sample_negative_video_feature(self, env_name):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == env_name:
            negative_env_name = random.choice(self.keys)

        negative_video_group = self.h5_file[negative_env_name]
        negative_datasets = list(negative_video_group.keys())
        negative_random_name = random.choice(negative_datasets)
        # negative_random_name = "0"
        negative_video_dataset = np.asarray(negative_video_group[negative_random_name])

        negative_start_idx = random.randint(0, len(negative_video_dataset)-7)
        negative_end_idx = random.randint(negative_start_idx+5, len(negative_video_dataset))

        negative_video_frames = np.array(negative_video_dataset)[negative_start_idx:negative_end_idx]
        if self.args.normalize_embedding:
            negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        
        if self.args.subsample_video:
            negative_video_frames = self.padding_video(negative_video_frames, self.args.max_length)
        video_progress = np.zeros(negative_video_frames.shape[0])

        return negative_video_frames, video_progress, np.zeros(negative_video_frames.shape[0])

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



    def sample_fully_reverse_video_feature(self, env_name):
        progress_group = self.h5_file[env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        # random_name = "0"
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        # only sample 2nd part of the video

        video_frames = np.array(progress_dataset)
        # second part of the video
        length = len(video_frames)
        second_part = video_frames[length//2:]
        second_part = second_part[::-1].copy()
        if len(second_part) > 10:
            start_idx = random.randint(0, len(second_part)-10)
            end_idx = random.randint(start_idx+10, len(second_part))
            second_part = second_part[start_idx:end_idx]

        if self.args.normalize_embedding:
            second_part = normalize_embeddings(second_part, return_tensor=True)
        # video_frames = video_frames.flip(dims=[0])
        

        if self.args.subsample_video:
            second_part = self.padding_video(second_part, self.args.max_length)

        progress = np.zeros(second_part.shape[0])

        return second_part, progress, np.zeros(second_part.shape[0])



    def sample_reverse_video_feature(self, env_name):
        progress_group = self.h5_file[env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        # random_name = "0"
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-7)
        end_idx = random.randint(start_idx+5, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        progress_idx= np.arange(0, video_frames.shape[0]) + 1
        progress = progress_idx / len(full_frames)

        if self.args.catagorical_progress:
            progress = np.floor(progress * self.args.catagorical_progress_bins) 
            if progress[-1] == self.args.catagorical_progress_bins:
                progress[-1] = self.args.catagorical_progress_bins - 1
            if self.args.sample_neg:
                progress += 1

        reverse_frame = video_frames[::-1][1:]
        reverse_progress = progress[::-1][1:]

        video_frames = np.concatenate([video_frames, reverse_frame], axis=0)
        progress = np.concatenate([progress, reverse_progress], axis=0)

        if self.args.normalize_embedding:
            video_frames = normalize_embeddings(video_frames, return_tensor=True)

        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)

            progress = np.expand_dims(progress, axis=1)
            progress = self.padding_video(progress, self.args.max_length).detach().cpu().numpy()
            progress = np.squeeze(progress, axis=1)

            return video_frames, progress, np.ones(video_frames.shape[0])
        else:
            return video_frames, progress, np.ones(video_frames.shape[0])
        

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
    class_label_output = list()
    

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
        class_label = th.tensor(batch[i]["class_label"])

        video_output.append(padded_video)
        mask_output.append(mask)
        text_output.append(text)
        progress_output.append(progress)
        class_label_output.append(class_label)


    output_dict = {
        "video_array": th.stack(video_output),
        "mask": th.stack(mask_output),
        "text_array": th.stack(text_output),
        "progress": th.stack(progress_output),
        "class_label": th.stack(class_label_output)
    }

    return output_dict


def video_collate_triangular_fn(batch):
    # Find the maximum video length (number of frames) in the batch

    length = [data["video_array"].shape[0] for data in batch]
    max_length = max(length)

    embedding_size = batch[0]["video_array"].shape[1]
    batch_size = len(batch)



    
    video_output = list()
    mask_output = list()
    text_output = list()
    progress_output = list()
    class_label_output = list()
    triangular_mask_output = list()
    
    
    for i in range(batch_size):
        video = batch[i]["video_array"]
        if type(video) == np.ndarray:
            video = th.tensor(video) 
        padding = th.zeros((max_length - video.shape[0], embedding_size))
        padded_video = th.cat((video, padding), dim=0)
        mask = th.zeros(max_length)
        mask[:video.shape[0]] = 1

        # generate triangular mask
        triangular_mask = th.zeros((max_length, max_length))
        for j in range(video.shape[0]):
            triangular_mask[j, :j+1] = 1
        
        padding_zeros = th.zeros(max_length - video.shape[0])
        progress = th.tensor(batch[i]["progress"])
        class_label = th.tensor(batch[i]["class_label"])

        text = th.tensor(batch[i]["text_array"])

        progress = th.cat([progress, padding_zeros], dim=0)
        
        class_label = class_label.repeat(max_length)
        

        video_output.append(padded_video)
        mask_output.append(mask)
        text_output.append(text)
        progress_output.append(progress)
        class_label_output.append(class_label)
        triangular_mask_output.append(triangular_mask)

    output_dict = {
        "video_array": th.stack(video_output),
        "mask": th.stack(mask_output),
        "text_array": th.stack(text_output),
        "progress": th.stack(progress_output),
        "class_label": th.stack(class_label_output),
        "triangular_mask": th.stack(triangular_mask_output)
    }

    return output_dict



