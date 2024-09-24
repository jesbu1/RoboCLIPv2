from torch.utils.data import Dataset, DataLoader
import h5py
from transformers import AutoTokenizer, AutoModel, AutoProcessor
# from s3dg import S3D
import torch as th
import random
import numpy as np
import copy
import json

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


class GifTextEmbeddingDataset(Dataset):
    def __init__(self, args):
        self.h5_file = h5py.File(args.h5_embedding_path, "r")
        self.h5_text_file = h5py.File(args.h5_text_path, "r")


        subset_list = json.load(open("task_subset.json"))
        if args.subset == 0:
        # if args.task_nums == 50:
            self.keys = list(self.h5_file["GT_Videos"].keys())
        else:
            subset_name = "subset_" + str(args.subset)
            self.keys = subset_list[subset_name]



    # evaluate_task = ["door-close-v2-goal-hidden", 
    #                 "door-open-v2-goal-hidden", 
    #                 "drawer-close-v2-goal-hidden", 
    #                 "button-press-v2-goal-hidden", 
    #                 "button-press-topdown-v2-goal-hidden"]


        self.shuffle_time = args.time_shuffle
        self.shorten_time = args.time_shorten
        self.random_sample_neg = args.rand_neg
        self.augmentation = args.augmentation

        self.candidate_type = [1] # 1: hard negative
        if args.time_shuffle:
            self.candidate_type.append(2)
        if args.time_shorten:
            self.candidate_type.append(3)
        if args.random_noise:
            self.candidate_type.append(4)

    def __len__(self):
        return len(self.keys)
        # return len(self.keys)


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        gt_array = np.asarray(self.h5_text_file[key])
        gt_array = np.expand_dims(gt_array, axis=0)

        pos_type = "regular"
        # sample positive sample
        if self.augmentation:
            pos_type = random.choice(["regular", "augment"])

        if pos_type == "regular":
            rand_idx = np.random.randint(0, 15)
            pos_array = self.h5_file["GT_Videos"][key][rand_idx]
        else:   
            rand_idx = np.random.randint(0, 15 * 80)
            pos_array = self.h5_file["Augmentation_Videos"][key][rand_idx]

        pos_array = np.expand_dims(pos_array, axis=0)

        # sample negative sample
        neg_type = random.choice(self.candidate_type)
        progress = 0

        if neg_type == 1: # need to find hardest negative in the main loop
            neg_array = np.random.normal(0, 1, pos_array.shape)
        elif neg_type == 2:
            # time shuffle
            neg_array = self.sample_shuffle_feature(key)
        elif neg_type == 3:
            # time shorten
            neg_array, progress = self.sample_shorten_feature(key)                
        elif neg_type == 4:
            # random noise: generate random gaussian noise with std 1 add on the pos array
            random_noise = np.random.normal(0, 1, pos_array.shape)
            neg_array = pos_array + random_noise * 0.1

        output_dict = {
            "gt_array": gt_array,
            "pos_array": pos_array,
            "neg_array": neg_array,
            "type": neg_type,
            "progress": progress,
        }

        return  output_dict


    def sample_shuffle_feature(self, env_name):
        # sample a time shuffle feature
        group = self.h5_file["Time_Shuffle_Videos"][env_name]
        random_idx = np.random.randint(0, len(group))
        neg_feature = group[random_idx][()]
        neg_feature = np.asarray(neg_feature)
        neg_feature = np.expand_dims(neg_feature, axis=0)
        return neg_feature

    
    def sample_shorten_feature(self, env_name):
        # sample a time shorten feature
        group_feature = self.h5_file["Time_Shortened_Videos"][env_name]
        group_progress = self.h5_file["Time_Shortened_Videos"][env_name + "_progress"]
        random_idx = np.random.randint(0, len(group_feature))
        neg_feature = np.asarray(group_feature[random_idx])
        progress = np.asarray(group_progress[random_idx])
        neg_feature = np.expand_dims(neg_feature, axis=0)
        return neg_feature, progress


    def __del__(self):
        self.h5_file.close()
        self.h5_text_file.close()
