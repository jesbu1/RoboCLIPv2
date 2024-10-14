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

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


class ClipLivDataset(Dataset):

    def __init__(self, args, h5_file):
        self.h5_file = h5_file
        subset_list = json.load(open("task_subset.json"))
        subset_name = "subset_6"
        self.keys = subset_list[subset_name]
        self.model_name = args.model_name
        self.sample_other_task = args.sample_other_task

    def __len__(self):
        return len(self.keys) * 1000


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)
        # sample video sample
        if self.sample_other_task:
            first_frame, last_frame, mid_frames_1, mid_frames_2, other_video_frame = self.sample_video_rank_feature(key)
            output_dict = {
                "text_array": text_array,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "mid_frames_1": mid_frames_1,
                "mid_frames_2": mid_frames_2,
                "other_video_frame": other_video_frame
            }
        else:
            first_frame, last_frame, mid_frames_1, mid_frames_2, _ = self.sample_video_rank_feature(key)
            output_dict = {
                "text_array": text_array,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "mid_frames_1": mid_frames_1,
                "mid_frames_2": mid_frames_2
            }


        return  output_dict
    
    def sample_text_feature(self, env_name):
        text_env_name = env_name + "_text"
        text_dataset = self.h5_file[self.model_name][text_env_name]
        # choose index
        idx = random.randint(0, len(text_dataset)-1)
        text_array = np.asarray(text_dataset[idx])
        return text_array

    def sample_video_rank_feature(self, env_name, sample_frames = 4):
        '''
        return first_frame, last_frame, mid_frames_1, mid_frames_2
        '''
        video_env_name = env_name
        video_group = self.h5_file[self.model_name][video_env_name]
        # choose index
        keys = list(video_group.keys())
        idx = random.randint(0, len(keys)-1)
        video_array = np.asarray(video_group[keys[idx]])

        first_frame = video_array[0]
        last_frame = video_array[-1]
        # random sample 2 idx, except the first and last frame
        idxs_1 = np.random.randint(1, len(video_array)-1, 1)
        # sample idx_2 != idx1
        idxs_2 = np.random.randint(1, len(video_array)-1, 1)
        while np.any(idxs_1 == idxs_2):
            idxs_2 = np.random.randint(1, len(video_array)-1, 1)

        other_video_frame = None
        if self.sample_other_task:
            other_env_name = random.choice(self.keys)
            while other_env_name == env_name:
                other_env_name = random.choice(self.keys)
            
            other_video_group = self.h5_file[self.model_name][other_env_name]
            other_keys = list(other_video_group.keys())
            other_idx = random.randint(0, len(other_keys)-1)
            other_video_array = np.asarray(video_group[other_keys[other_idx]])
            frame_idx = np.random.randint(0, len(other_video_array), 1)
            other_video_frame = other_video_array[frame_idx].squeeze()
        if idxs_1 < idxs_2:
            return first_frame, last_frame, video_array[idxs_1].squeeze(), video_array[idxs_2].squeeze(), other_video_frame
        else:
            return first_frame, last_frame, video_array[idxs_2].squeeze(), video_array[idxs_1].squeeze(), other_video_frame

    def __del__(self):
        self.h5_file.close()


class ClipLivProgressDataset(ClipLivDataset):

    def __len__(self):
        return len(self.keys) * 1000


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)
        progress_array, progress, goal_array = self.sample_progress_feature(key)
        # sample video sample
        return  {
            "text_array": text_array,
            "progress_array": progress_array,
            "progress": progress,
            "goal_array": goal_array
        }


    def sample_progress_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        # choose index
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name])
        idx = random.randint(0, len(progress_dataset)-1)
        video_array = np.asarray(progress_dataset[idx])
        goal_array = progress_dataset[-1]
        
        length = len(progress_dataset)
        progress = (idx + 1) / length

        return video_array, progress, goal_array



class ClipLivSingleDataset(Dataset):

    def __init__(self, args, h5_file):
        self.h5_file = h5_file
        subset_list = json.load(open("task_subset.json"))
        subset_name = "subset_6"
        self.keys = subset_list[subset_name]
        self.model_name = args.model_name
        if args.loss_type == "mse":
            self.sample_goal = False
        else:
            self.sample_goal = True

    def __len__(self):
        return len(self.keys) * 1000


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)
        # sample video sample
        video_array, progress = self.sample_progress_feature(key)
        if self.sample_goal:
            goal_array = self.sample_goal_feature(key)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
        }

        if self.sample_goal:
            output_dict["goal_array"] = goal_array

        return  output_dict
    
    def sample_text_feature(self, env_name):
        text_env_name = env_name + "_text"
        text_dataset = self.h5_file[self.model_name][text_env_name]
        # choose index
        idx = random.randint(0, len(text_dataset)-1)
        text_array = np.asarray(text_dataset[idx])
        return text_array


    def sample_progress_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        # choose index
        datasets = sorted(list(progress_group.keys()), key = int)[:15]
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name])
        idx = random.randint(0, len(progress_dataset)-1)
        video_array = np.asarray(progress_dataset[idx])
        
        length = len(progress_dataset)
        progress = (idx + 1) / length

        return video_array, progress

    def sample_goal_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        # choose index
        datasets = sorted(list(progress_group.keys()), key = int)[:15]
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name])
        goal_array = progress_dataset[-1]

        return goal_array
    

    def __del__(self):
        self.h5_file.close()












# if __name__ == "__main__":
#     a = 0
    # args = {
    #     "h5_embedding_path": "metaworld_25_for_clip_liv.h5",
    #     "subset": 0,
    #     "time_shuffle": True,
    #     "time_shorten": True,
    #     "rand_neg": True,
    #     "augmentation": True
    # }
    # args = argparse.Namespace(**args)
    # dataset = ClipLivDataset(args)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for data in dataloader:
    #     print(data["gt_array"].shape, data["pos_array"].shape, data["neg_array"].shape)
    #     break