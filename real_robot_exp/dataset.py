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
import json

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings).float()
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


class LivRealVideoDataset(Dataset):

    def __init__(self, args, h5_file, split=False):
        h5_file = h5py.File(h5_file, "r")
        self.h5_file = h5_file
        self.args = args
        self.split = split
        self.keys = list(self.h5_file.keys())
        if self.split:
            self.keys = self.keys[:int(len(self.keys)*0.5)]
            eval_keys = self.keys[int(len(self.keys)*0.5):]
            json.dump(eval_keys, open("eval_keys.json", "w"), indent=4)


    def __len__(self):
        if self.split:
            
            return 3200
        return 6400
    
    def __getitem__(self, idx):
        # select a random key
        key_id = random.randint(0, len(self.keys)-1)
        key = self.keys[key_id]
        data_group = self.h5_file[key]

        # sample text sample
        text_array = self.sample_text_feature(data_group)

        if not self.split:
            if self.args.sample_neg:
                if random.random() > 0.75:
                    video_array, progress, class_label = self.sample_negative_video_feature(key)
                else:
                    video_array, progress, class_label = self.sample_video_feature(data_group)
            else:
                video_array, progress, class_label = self.sample_video_feature(data_group)
        else:
            if self.args.sample_neg:
                if self.args.reverse_video:
                    random_num = random.random()
                    if random_num < 0.35:
                        video_array, progress, class_label = self.sample_negative_video_feature(key)
                    elif random_num > 0.50:
                        video_array, progress, class_label = self.sample_reverse_video_feature(data_group)
                    else:
                        video_array, progress, class_label = self.sample_video_feature(data_group)
                else:
                    random_num = random.random()
                    if random_num < 0.35:
                        video_array, progress, class_label = self.sample_negative_video_feature(key)
                    else:
                        video_array, progress, class_label = self.sample_video_feature(data_group)
            else:
                video_array, progress, class_label = self.sample_video_feature(data_group)







        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "class_label": class_label
        }
        return  output_dict

    def sample_text_feature(self, data_group):
        lang_embedding = np.array(data_group["lang_embedding"])
        if lang_embedding.shape[0] == 1024:
            lang_embedding = np.expand_dims(lang_embedding, axis=0)
        if self.args.normalize_embedding:
            lang_embedding = normalize_embeddings(lang_embedding, return_tensor=True)

        return lang_embedding
    
    def sample_video_feature(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists.remove("lang_embedding")
        random_name = random.choice(traj_lists)

        progress_dataset = np.asarray(data_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-3)
        end_idx = random.randint(start_idx+3, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        if self.args.normalize_embedding:
            video_frames = normalize_embeddings(video_frames, return_tensor=True)
        else:
            video_frames = th.tensor(video_frames)
        full_length = len(full_frames)
        video_progress = np.arange(0, video_frames.shape[0]) + 1
        video_progress = video_progress / full_length

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

        return video_frames, video_progress, np.ones(video_progress.shape[0])

    def sample_negative_video_feature(self, env_name):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == env_name:
            negative_env_name = random.choice(self.keys)

        negative_video_group = self.h5_file[negative_env_name]
        negative_datasets = list(negative_video_group.keys())
        negative_datasets.remove("lang_embedding")
        negative_random_name = random.choice(negative_datasets)
        negative_video_dataset = np.asarray(negative_video_group[negative_random_name])

        negative_start_idx = random.randint(0, len(negative_video_dataset)-3)
        negative_end_idx = random.randint(negative_start_idx+3, len(negative_video_dataset))

        negative_video_frames = np.array(negative_video_dataset)[negative_start_idx:negative_end_idx]
        if self.args.normalize_embedding:
            negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        
        if self.args.subsample_video:
            negative_video_frames = self.padding_video(negative_video_frames, self.args.max_length)
        video_progress = np.zeros(negative_video_frames.shape[0])

        return negative_video_frames, video_progress, np.zeros(video_progress.shape[0])


    def sample_reverse_video_feature(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists.remove("lang_embedding")
        random_name = random.choice(traj_lists)

        progress_dataset = np.asarray(data_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-3)
        end_idx = random.randint(start_idx+3, len(progress_dataset))

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
            return video_frames, progress, np.ones(progress.shape[0])
        else:
            return video_frames, progress, np.ones(progress.shape[0])
        




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


        