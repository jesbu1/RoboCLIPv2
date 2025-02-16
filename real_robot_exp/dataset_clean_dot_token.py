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
    if len(embeddings.shape) == 1:
        embeddings = embeddings.unsqueeze(0)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()



class LivRealVideoTrainTokenDataset(Dataset):


    def __init__(self, args, h5_file, split=False, eval=False, sample_neg=False):
        h5_file = h5py.File(h5_file, "r")
        self.h5_file = h5_file
        self.args = args
        self.keys = list(self.h5_file.keys())

        self.sample_neg = sample_neg




    def sample_text_feature(self, data_group):
        key = list(data_group.keys())
        key = [k for k in key if "liv_lang_embedding_individual" in k]
        # if len(key) > 1:
        #     key.remove("lang_embedding_individual")
        
        key = random.choice(key)
        lang_embedding = np.array(data_group[key])[0]

        if self.args.normalize_embedding:
            lang_embedding = normalize_embeddings(lang_embedding, return_tensor=True)
   
        return lang_embedding
    

    def sample_negative_text_feature(self, key):
        random_key = random.choice(self.keys)
        
        while random_key == key:
            random_key = random.choice(self.keys)
        
        data_group = self.h5_file[random_key]
        key = list(data_group.keys())
        key = [k for k in key if "liv_lang_embedding_individual" in k]
        
        # if len(key) > 1:
        #     key.remove("lang_embedding_individual")
        key = random.choice(key)
        lang_embedding = np.array(data_group[key])[0]

        if self.args.normalize_embedding:
            lang_embedding = normalize_embeddings(lang_embedding, return_tensor=True).squeeze(0)

        return lang_embedding


    def sample_video_feature(self, data_group):

        traj_lists = list(data_group.keys())

        traj_lists = [traj for traj in traj_lists if "lang" not in traj]  

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

        return video_frames, video_progress, np.ones(video_progress.shape[0])



    def sample_reverse_video_feature(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists = [traj for traj in traj_lists if "lang" not in traj]   
        random_name = random.choice(traj_lists)

        progress_dataset = np.asarray(data_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-3)
        end_idx = random.randint(start_idx+3, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        progress_idx= np.arange(0, video_frames.shape[0]) + 1
        progress = progress_idx / len(full_frames)

        # rewind the video
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
    
    def __len__(self):
        # if self.split:
        #     return self.args.batch_size * 100

        return int(self.args.batch_size * 100 * (1 - self.args.extra_data_ratio)) + 1


    def __getitem__(self, idx):
        # select a random key
        key_id = random.randint(0, len(self.keys)-1)
        key = self.keys[key_id]
        data_group = self.h5_file[key]

        if self.args.rewind:
            random_num = random.random()
            if random_num < 0.5:
                video_array, progress, class_label = self.sample_reverse_video_feature(data_group)
            else:
                video_array, progress, class_label = self.sample_video_feature(data_group)
        else:
            video_array, progress, class_label = self.sample_video_feature(data_group)

        # sample text sample
        if self.sample_neg:
            if random.random() < 0.2:
                text_array = self.sample_negative_text_feature(key)
                progress = np.zeros(progress.shape)
                class_label = np.zeros(class_label.shape)
            else:
                text_array = self.sample_text_feature(data_group)
        else:
            text_array = self.sample_text_feature(data_group)


        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "class_label": class_label
        }
        return  output_dict
    



class LivRealVideoEvalTokenDataset(Dataset):

    def __init__(self, args, h5_file, label="positive"):
        self.h5_file = h5_file
        self.args = args
        self.label = label
        self.keys = list(self.h5_file.keys())

    def __len__(self):

        return len(self.keys)
    
    def __getitem__(self, idx):


        key_id = idx % len(self.keys)
        key = self.keys[key_id]
        data_group = self.h5_file[key]

        # sample text sample
        text_array = self.sample_text_feature(data_group)

        if self.label == "positive":
            video_array, progress, class_label = self.sample_video_feature(data_group)
        else:
            video_array, progress, class_label = self.sample_negative_video_feature(key)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "class_label": class_label
        }
        return  output_dict


    def sample_text_feature(self, data_group):
        key = list(data_group.keys())
        key = [k for k in key if "liv_lang_embedding_individual" in k]
        
        key = random.choice(key)
        lang_embedding = np.array(data_group[key])[0]

        if self.args.normalize_embedding:
            lang_embedding = normalize_embeddings(lang_embedding, return_tensor=True)
   
        return lang_embedding
    
    

    def sample_video_feature(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists = [traj for traj in traj_lists if "lang" not in traj]   
        random_name = random.choice(traj_lists)

        video_frames = np.asarray(data_group[random_name]) # all video data

        if self.args.normalize_embedding:
            video_frames = normalize_embeddings(video_frames, return_tensor=True)
        else:
            video_frames = th.tensor(video_frames)
        full_length = len(video_frames)
        video_progress = np.arange(0, video_frames.shape[0]) + 1
        video_progress = video_progress / full_length
        if self.args.catagorical_progress:
            video_progress = np.floor(video_progress * self.args.catagorical_progress_bins) 
            if video_progress[-1] == self.args.catagorical_progress_bins:
                video_progress[-1] = self.args.catagorical_progress_bins - 1
            if not self.args.two_step_training:
                video_progress += 1

        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)
            video_progress = np.expand_dims(video_progress, axis=1)
            video_progress = self.padding_video(video_progress, self.args.max_length).detach().cpu().numpy()
            video_progress = np.squeeze(video_progress, axis=1)


        return video_frames, video_progress, np.ones(video_progress.shape[0])

    def sample_negative_video_feature(self, env_name):

        negative_env_name = random.choice(self.keys)
        while negative_env_name == env_name:
            negative_env_name = random.choice(self.keys)

        negative_video_group = self.h5_file[negative_env_name]
        negative_datasets = list(negative_video_group.keys())
        negative_datasets = [dataset for dataset in negative_datasets if "lang" not in dataset]

        negative_random_name = random.choice(negative_datasets)
        negative_video_frames = np.asarray(negative_video_group[negative_random_name])

        if self.args.normalize_embedding:
            negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        
        if self.args.subsample_video:
            negative_video_frames = self.padding_video(negative_video_frames, self.args.max_length)
        video_progress = np.zeros(negative_video_frames.shape[0])

        return negative_video_frames, video_progress, np.zeros(video_progress.shape[0])



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


def VideoTextTokenCollateFn(batch):
    # Find the maximum video length (number of frames) in the batch

    length = [data["text_array"].shape[0] for data in batch]
    max_length = max(length)
    embedding_size = batch[0]["text_array"].shape[1]
    batch_size = len(batch)

    video_feature_output = list()
    text_feature_output = list()
    text_mask_output = list()
    progress_output = list()
    class_label_output = list()

    for i in range(batch_size):

        text_feature = batch[i]["text_array"]
        video_feature = batch[i]["video_array"]
        progress = batch[i]["progress"]
        class_label = batch[i]["class_label"]
        feature_length = text_feature.shape[0]
        padding_length = max_length - feature_length
        
        if padding_length != 0:
            text_feature_padding = np.zeros((padding_length, embedding_size))
            text_feature = np.concatenate([text_feature, text_feature_padding], axis=0)
            text_mask = np.ones((max_length))
            text_mask[:feature_length] = 0
        else:
            text_mask = np.zeros((max_length))

        video_feature_output.append(video_feature)
        text_feature_output.append(text_feature)
        text_mask_output.append(text_mask)
        progress_output.append(progress)
        class_label_output.append(class_label)


    video_feature_output = np.stack(video_feature_output, axis=0)
    text_feature_output = np.stack(text_feature_output, axis=0)
    text_mask_output = np.stack(text_mask_output, axis=0)
    progress_output = np.stack(progress_output, axis=0)
    class_label_output = np.stack(class_label_output, axis=0)

    output_dict = {
        "video_array": th.tensor(video_feature_output),
        "text_array": th.tensor(text_feature_output),
        "text_mask": th.tensor(text_mask_output),
        "progress": th.tensor(progress_output).float(),
        "class_label": th.tensor(class_label_output)
    }

    return output_dict