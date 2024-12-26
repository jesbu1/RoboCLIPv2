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
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(progress_dataset) - start_idx
        progress = video_frames.shape[0] / length

        if self.args.random_shuffle:
            if random.random() > 0.667:
                video_frames = video_frames[th.randperm(video_frames.size(0))]
                progress = 0

        return video_frames, progress


class ClipLivVideoSameLengthDataset(ClipLivVideoDataset):

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(video_frames)
        start_length = len(progress_dataset) - start_idx
        progress = length / start_length

        if len(video_frames) < self.args.sample_frames:
            # padding 1st frame
            num_pads = self.args.sample_frames - length
            first_frame = video_frames[0:1]
            video_frames = th.cat([first_frame]*num_pads + [video_frames], dim=0)

        elif len(video_frames) > self.args.sample_frames:

            # sample frames
            first_frame = video_frames[0:1]
            last_frame = video_frames[-1:]
            mid_frames = video_frames[1:-1]
            num_samples = self.args.sample_frames - 2
            frame_idx = th.linspace(0, len(mid_frames) - 1, num_samples, dtype=th.long)
            # even sample from frame_idx
            # same stride sample
            choose_idx = random.sample(list(frame_idx), num_samples)
            choose_idx = th.sort(th.tensor(choose_idx))[0]

            # all_idx = list(range(1, len(mid_frames)))
            # choose_idx = all_idx[::len(all_idx)//num_samples]

            mid_frames = mid_frames[choose_idx]
            video_frames = th.cat([first_frame, mid_frames, last_frame], dim=0)


        return video_frames, progress



class ClipLivVideoReverseDataset(ClipLivVideoDataset):

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

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
        end_idx = random.randint(start_idx+1, len(progress_dataset))

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




class ClipLivVideoClassDataset(Dataset):

    def __init__(self, args, h5_file):
        self.h5_file = h5_file
        subset_list = json.load(open("task_subset.json"))
        subset_name = "subset_6"
        self.keys = subset_list[subset_name]
        self.model_name = args.model_name
        self.args = args
        self.sample_negative = args.sample_negative
        self.num_class = args.num_class

    def sample_negative_video_feature(self, env_name):
        # sample a name not same as env_name
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
        negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        class_label = 0

        return negative_video_frames, class_label


    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(progress_dataset) - start_idx
        progress = video_frames.shape[0] / length
        
        class_label = progress * self.num_class
        class_label = int(class_label)
        # if progress >= 1:
            # print("progress", progress, "class_label", class_label, "start_idx", start_idx, "end_idx", end_idx, "frame shape", video_frames.shape, "length", length, "dataset", len(progress_dataset))
        if progress == 1:
            class_label -= 1
        if self.sample_negative:
            class_label += 1
        
        return video_frames, class_label

    def __len__(self):
        return len(self.keys) * 1000

    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)

        if self.sample_negative:
            if random.random() > 0.25:
                video_array, class_label = self.sample_progress_video_feature(key)
            else:
                video_array, class_label = self.sample_negative_video_feature(key)
        else:
            video_array, class_label = self.sample_progress_video_feature(key)


        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "class_label": class_label
        }

        return  output_dict

    def sample_text_feature(self, env_name):
        text_env_name = env_name + "_text"
        text_dataset = self.h5_file[self.model_name][text_env_name]
        # choose index
        idx = random.randint(0, len(text_dataset)-1)
        text_array = np.asarray(text_dataset[idx])
        return text_array





class ClipLivVideoNegDataset(ClipLivVideoDataset):

    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)

        if random.random() > 0.25:
            video_array, progress = self.sample_progress_video_feature(key)
        else:
            video_array, progress = self.sample_negative_video_feature(key)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress
        }

        return  output_dict

    def sample_negative_video_feature(self, env_name):
        # sample a name not same as env_name
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
        negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        progress = 0

        return negative_video_frames, progress


    


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





class ClipLivVideoSameLengthNegDataset(ClipLivVideoDataset):

    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(video_frames)
        start_length = len(progress_dataset) - start_idx
        progress = length / start_length

        if len(video_frames) < self.args.sample_frames:
            # padding 1st frame
            num_pads = self.args.sample_frames - length
            first_frame = video_frames[0:1]
            video_frames = th.cat([first_frame]*num_pads + [video_frames], dim=0)

        elif len(video_frames) > self.args.sample_frames:

            # sample frames
            first_frame = video_frames[0:1]
            last_frame = video_frames[-1:]
            mid_frames = video_frames[1:-1]
            num_samples = self.args.sample_frames - 2
            frame_idx = th.linspace(0, len(mid_frames) - 1, num_samples, dtype=th.long)
            # even sample from frame_idx
            # same stride sample
            choose_idx = random.sample(list(frame_idx), num_samples)
            choose_idx = th.sort(th.tensor(choose_idx))[0]

            # all_idx = list(range(1, len(mid_frames)))
            # choose_idx = all_idx[::len(all_idx)//num_samples]

            mid_frames = mid_frames[choose_idx]
            video_frames = th.cat([first_frame, mid_frames, last_frame], dim=0)

        if self.args.random_shuffle:
            if random.random() > 0.667:
                video_frames = video_frames[th.randperm(video_frames.size(0))]
                progress = 0

        return video_frames, progress

    def sample_negative_video_feature(self, env_name):
        # sample a name not same as env_name
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
        # sample 32 frames
        if len(negative_video_frames) < self.args.sample_frames:
            # padding 1st frame
            num_pads = self.args.sample_frames - len(negative_video_frames)
            first_frame = negative_video_frames[0:1]
            # negative_video_frames = th.cat([first_frame]*num_pads + [negative_video_frames], dim=0)
            negative_video_frames = np.concatenate([first_frame]*num_pads + [negative_video_frames], axis=0)
        else:
            # sample frames
            first_frame = negative_video_frames[0:1]
            last_frame = negative_video_frames[-1:]
            mid_frames = negative_video_frames[1:-1]
            num_samples = self.args.sample_frames - 2
            frame_idx = th.linspace(0, len(mid_frames) - 1, num_samples, dtype=th.long)
            # even sample from frame_idx
            choose_idx = random.sample(list(frame_idx), num_samples)
            choose_idx = th.sort(th.tensor(choose_idx))[0]


            mid_frames = mid_frames[choose_idx]
            negative_video_frames = np.concatenate([first_frame, mid_frames, last_frame], axis=0)

        negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        progress = 0

        return negative_video_frames, progress

    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)

        if self.args.sample_negative:
            if random.random() > 0.25:
                
                video_array, progress = self.sample_progress_video_feature(key)
            else:
                video_array, progress = self.sample_negative_video_feature(key)

        else:
            video_array, progress = self.sample_progress_video_feature(key)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress
        }

        return  output_dict






def video_class_collate_fn(batch):
    # Find the maximum video length (number of frames) in the batch

    length = [data["video_array"].shape[0] for data in batch]
    max_length = max(length)

    embedding_size = batch[0]["video_array"].shape[1]
    batch_size = len(batch)

    
    video_output = list()
    mask_output = list()
    text_output = list()
    class_output = list()

    for i in range(batch_size):
        video = batch[i]["video_array"]
        padding = th.zeros((max_length - video.shape[0], embedding_size))
        padded_video = th.cat((video, padding), dim=0)
        mask = th.zeros(max_length)
        mask[:video.shape[0]] = 1
        text = th.tensor(batch[i]["text_array"])
        class_label = th.tensor(batch[i]["class_label"])

        video_output.append(padded_video)
        mask_output.append(mask)
        text_output.append(text)
        class_output.append(class_label)

    output_dict = {
        "video_array": th.stack(video_output),
        "mask": th.stack(mask_output),
        "text_array": th.stack(text_output),
        "class_output": th.stack(class_output)
    }

    return output_dict



class ClipLivVideoNegClassDataset(ClipLivVideoDataset):
    # this method include 2 things
    '''
    1. if this text and video are from same class
    2. if this video is random shuffle
    3. video progress

    same class: 0
    shuffle: 1
    different class: 2

    '''

    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        
        # sample text sample
        text_array = self.sample_text_feature(key)

        if random.random() > 0.25:
            video_array, progress, label = self.sample_progress_video_feature(key)
        else:
            video_array, progress, label = self.sample_negative_video_feature(key)

        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "class_label": label
        }
        return  output_dict


    def sample_progress_video_feature(self, env_name):
        progress_group = self.h5_file[self.model_name][env_name]
        datasets = list(progress_group.keys())
        random_name = random.choice(datasets)
        progress_dataset = np.asarray(progress_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)-2)
        end_idx = random.randint(start_idx+1, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        video_frames = normalize_embeddings(video_frames, return_tensor=True)
        length = len(progress_dataset) - start_idx
        progress = video_frames.shape[0] / length

        label = 0
        if self.args.random_shuffle:
            if random.random() > 0.667:
                video_frames = video_frames[th.randperm(video_frames.size(0))]
                progress = 0
                label = 2

        return video_frames, progress, label


    def sample_negative_video_feature(self, env_name):
        # sample a name not same as env_name
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
        negative_video_frames = normalize_embeddings(negative_video_frames, return_tensor=True)

        progress = 0
        label = 1

        return negative_video_frames, progress, label


def video_collate_fn(batch):
    # Find the maximum video length (number of frames) in the batch

    length = [data["video_array"].shape[0] for data in batch]
    max_length = max(length)

    embedding_size = batch[0]["video_array"].shape[1]
    batch_size = len(batch)

    
    video_output = list()
    mask_output = list()
    text_output = list()
    label_output = list()
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
        label = th.tensor(batch[i]["class_label"])

        video_output.append(padded_video)
        mask_output.append(mask)
        text_output.append(text)
        progress_output.append(progress)
        label_output.append(label)


    output_dict = {
        "video_array": th.stack(video_output),
        "mask": th.stack(mask_output),
        "text_array": th.stack(text_output),
        "progress": th.stack(progress_output),
        "class_label": th.stack(label_output)
    }

    return output_dict

