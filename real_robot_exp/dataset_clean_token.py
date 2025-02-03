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
import math

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings).float()
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
        self.split = split
        self.keys = list(self.h5_file.keys())
        if self.split:
            if eval:
                self.keys = self.keys[int(len(self.keys)*0.5):]
            else:
                self.keys = self.keys[:int(len(self.keys)*0.5)]
            eval_keys = self.keys[int(len(self.keys)*0.5):]
            json.dump(eval_keys, open("eval_keys.json", "w"), indent=4)
        self.sample_neg = sample_neg

        if self.args.text_positional_encoding:
            self.text_positional_encoding = self.get_cosine_positional_encoding(70, 1024).detach().cpu().numpy()
        if self.args.video_positional_encoding:
            self.video_positional_encoding = self.get_cosine_positional_encoding(self.args.max_length, 1024).detach().cpu().numpy()





    def get_cosine_positional_encoding(self, max_seq_len, embed_dim):
        """
        Generate a static positional encoding matrix using sine and cosine functions.
        """
        position = th.arange(max_seq_len).unsqueeze(1)  # Shape: [max_seq_len, 1]
        div_term = th.exp(th.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = th.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = th.sin(position * div_term)  # Even indices
        pe[:, 1::2] = th.cos(position * div_term)  # Odd indices
        pe /= 10 # reduce the scale of positional encoding otherwise it will dominate the input embeddings
        return pe


    def sample_text_feature(self, data_group):
        lang_embedding = np.array(data_group["lang_embedding_individual"])

        if lang_embedding.shape[0] == 1024:
            lang_embedding = np.expand_dims(lang_embedding, axis=0)

        len_lang_embedding = lang_embedding.shape[0]
        if len_lang_embedding > 1:
            lang_embedding = lang_embedding[random.randint(0, len_lang_embedding-1)]
        else:
            lang_embedding = lang_embedding[0]

        if self.args.normalize_embedding:
            lang_embedding = normalize_embeddings(lang_embedding, return_tensor=True).squeeze(0)
        if self.args.text_positional_encoding:
            lang_embedding = lang_embedding + self.text_positional_encoding[:lang_embedding.shape[0]]     

        return lang_embedding
    

    def sample_negative_text_feature(self, key):
        random_key = random.choice(self.keys)
        while random_key == key:
            random_key = random.choice(self.keys)
        data_group = self.h5_file[random_key]
        lang_embedding = np.array(data_group["lang_embedding_individual"])

        if lang_embedding.shape[0] == 1024:
            lang_embedding = np.expand_dims(lang_embedding, axis=0)

        len_lang_embedding = lang_embedding.shape[0]
        if len_lang_embedding > 1:
            lang_embedding = lang_embedding[random.randint(0, len_lang_embedding-1)]
        else:
            lang_embedding = lang_embedding[0]

        if self.args.normalize_embedding:
            lang_embedding = normalize_embeddings(lang_embedding, return_tensor=True).squeeze(0)

        if self.args.text_positional_encoding:
            lang_embedding = lang_embedding + self.text_positional_encoding[:lang_embedding.shape[0]] 
        return lang_embedding


    def sample_video_feature(self, data_group):

        traj_lists = list(data_group.keys())
        traj_lists.remove("lang_embedding")
        traj_lists.remove("lang_embedding_individual")  
        
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
            if not self.args.two_step_training:
                video_progress += 1

        if self.args.video_positional_encoding:
            video_frames = video_frames + self.video_positional_encoding

        return video_frames, video_progress, np.ones(video_progress.shape[0])



    def sample_reverse_video_feature(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists.remove("lang_embedding")
        if "lang_embedding_individual" in traj_lists:
            traj_lists.remove("lang_embedding_individual")  
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
            if not self.args.two_step_training:
                progress += 1

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

            if self.args.video_positional_encoding:
                video_frames = video_frames + self.video_positional_encoding
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
        if self.split:
            
            return self.args.batch_size * 100
        return self.args.batch_size * 100 * 3


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
            if random.random() < 0.5:
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


def VideoTextTokenCollateFn(batch):
    # Find the maximum video length (number of frames) in the batch

    length = [data["text_array"].shape[0] for data in batch]
    max_length = max(length)
    embedding_size = batch[0]["text_array"].shape[1]
    batch_size = len(batch)

    total_max_length = max_length + batch[0]["video_array"].shape[0]


    feature_output = list()
    text_mask_output = list()
    video_mask_output = list()
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
            feature_padding = np.zeros((padding_length, embedding_size))
            feature = np.concatenate([text_feature, video_feature, feature_padding], axis=0)

            text_mask = np.zeros((total_max_length))
            text_mask[:feature_length] = 1
            video_mask = np.zeros((total_max_length))
            video_mask[feature_length:feature_length+video_feature.shape[0]] = 1

            feature_output.append(feature)
            text_mask_output.append(text_mask)
            video_mask_output.append(video_mask)
            progress_output.append(progress)
            class_label_output.append(class_label)

        else:
            feature = np.concatenate([text_feature, video_feature], axis=0)
            text_mask = np.zeros((total_max_length))
            text_mask[:feature_length] = 1
            video_mask = np.zeros((total_max_length))
            video_mask[feature_length:feature_length+video_feature.shape[0]] = 1

            feature_output.append(feature)
            text_mask_output.append(text_mask)
            video_mask_output.append(video_mask)
            progress_output.append(progress)
            class_label_output.append(class_label)

    feature_output = np.stack(feature_output, axis=0)
    text_mask_output = np.stack(text_mask_output, axis=0)
    video_mask_output = np.stack(video_mask_output, axis=0)
    progress_output = np.stack(progress_output, axis=0)
    class_label_output = np.stack(class_label_output, axis=0)

    output_dict = {
        "feature_array": th.tensor(feature_output).float(),
        "text_mask": th.tensor(text_mask_output),
        "video_mask": th.tensor(video_mask_output),
        "progress": th.tensor(progress_output).float(),
        "class_label": th.tensor(class_label_output)
    }

    return output_dict



class LivRealVideoTokenEvalDataset(LivRealVideoTrainTokenDataset):

    def __init__(self, args, h5_file, split=False, eval=True, positive=True):
        h5_file = h5py.File(h5_file, "r")
        self.h5_file = h5_file
        self.args = args
        self.split = split
        self.keys = list(self.h5_file.keys())
        if self.split:
            if eval:
                self.keys = self.keys[int(len(self.keys)*0.5):]
            else:
                self.keys = self.keys[:int(len(self.keys)*0.5)]
            eval_keys = self.keys[int(len(self.keys)*0.5):]
        self.keys = eval_keys
        self.positive = positive

    def __len__(self):
        return len(self.keys)

    def sample_full_video_feature(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists.remove("lang_embedding")
        traj_lists.remove("lang_embedding_individual")  
        
        random_name = random.choice(traj_lists)

        progress_dataset = np.asarray(data_group[random_name])
        full_frames = np.array(progress_dataset)
        if self.args.normalize_embedding:
            full_frames = normalize_embeddings(full_frames, return_tensor=True)
        else:
            full_frames = th.tensor(full_frames)

        full_length = len(full_frames)
        progress = np.arange(0, full_length) + 1
        progress = progress / full_length

        if self.args.subsample_video:
            full_frames = self.padding_video(full_frames, self.args.max_length)
            progress = np.expand_dims(progress, axis=1)
            progress = self.padding_video(progress, self.args.max_length).detach().cpu().numpy()
            progress = np.squeeze(progress, axis=1)

        if self.args.catagorical_progress:
            progress = np.floor(progress * self.args.catagorical_progress_bins) 
            if progress[-1] == self.args.catagorical_progress_bins:
                progress[-1] = self.args.catagorical_progress_bins - 1
            if not self.args.two_step_training:
                progress += 1

        return full_frames, progress, np.ones(progress.shape[0])



    def __getitem__(self, idx):
        key = self.keys[idx]
        data_group = self.h5_file[key]

        video_array, progress, class_label = self.sample_full_video_feature(data_group)
        if self.positive:
            text_array = self.sample_text_feature(data_group)
        else:
            text_array = self.sample_negative_text_feature(key)

        if not self.positive:
            progress = np.zeros(progress.shape)
            class_label = np.zeros(class_label.shape)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "class_label": class_label
        }
        return  output_dict
