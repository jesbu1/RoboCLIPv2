# dataset and dataloader
from torch.utils.data import Dataset, DataLoader
import h5py
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from s3dg import S3D
import torch as th
import random
import numpy as np
import copy

def load_model(model_name):
    if model_name == "xclip":
        model_name = "microsoft/xclip-base-patch16-zero-shot"
        xclip_tokenizer = AutoTokenizer.from_pretrained(model_name)
        xclip_net = AutoModel.from_pretrained(model_name)
        xclip_processor = AutoProcessor.from_pretrained(model_name)
        return xclip_tokenizer, xclip_net, xclip_processor
    elif model_name == "s3d":
        s3d_model = S3D()
        s3d_model.load_state_dict(th.load('s3d_howto100m.pth'))
        return s3d_model
    else:
        raise ValueError("Model name not found")


class GifDataset(Dataset):
    def __init__(self, args):
        self.h5_file = h5py.File(args.h5_path, "r")
        self.keys = list(self.h5_file.keys())
        self.keys = ['door-close-v2-goal-hidden']


        self.vlm_name = args.model_name
        # if self.vlm_name == "xclip":
            # self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
            # self.model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").cuda()
        # else:
        #     self.s3d_model = load_model("s3d")  

        self.shuffle_time = args.time_shuffle
        self.shorten_time = args.time_shorten

    def __len__(self):
        return len(self.keys) * 20


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        group = self.h5_file[key]
        gif_names = list(group.keys())

        # sample gt sample
        gt_gif_name = random.choice(gif_names)
        gt_array = group[gt_gif_name][()]

        # sample positive sample
        pos_gif_name = random.choice(gif_names)
        pos_array = group[pos_gif_name][()]

        # sample negative sample
        if self.shuffle_time or self.shorten_time:
            # choose shuffle time or sample negative sample from other key, choose 1 from 2
            if random.random() < 0.5:
                if self.shuffle_time and self.shorten_time:
                    # choose shuffle time or shorten time, choose 1 from 2
                    if random.random() < 0.5:
                        # shuffle time
                        neg_array = self.shuffle_time_func(gt_array.copy())
                    else:
                        # shorten time
                        neg_array = self.shorten_time_func(gt_array.copy())
                elif self.shuffle_time:
                    # shuffle time
                    neg_array = self.shuffle_time_func(gt_array.copy())
                else:
                    # shorten time
                    neg_array = self.shorten_time_func(gt_array)

            else:
                # sample negative sample from other key
                neg_array = self.sample_negative_func(key)

        else:
            # sample negative sample from other key
            neg_array = self.sample_negative_func(key)

        # sample frames
        gt_array = self.sample_frames(gt_array)
        pos_array = self.sample_frames(pos_array)
        neg_array = self.sample_frames(neg_array)

        # preprocess
        if self.vlm_name == "xclip":

            gt_array = self.preprocess_xclip(gt_array)
            pos_array = self.preprocess_xclip(pos_array)
            neg_array = self.preprocess_xclip(neg_array)

        else:
            gt_array = gt_array/255
            pos_array = pos_array/255
            neg_array = neg_array/255

            gt_array = gt_array.transpose(3, 0, 1, 2)
            pos_array = pos_array.transpose(3, 0, 1, 2)
            neg_array = neg_array.transpose(3, 0, 1, 2)

            gt_array = th.from_numpy(gt_array).float()
            pos_array = th.from_numpy(pos_array).float()
            neg_array = th.from_numpy(neg_array).float()

        output_dict = {
            "gt_array": gt_array,
            "pos_array": pos_array,
            "neg_array": neg_array
        }

        return  output_dict


    def shuffle_time_func(self, array):
        random_index = np.random.permutation(len(array))
        return array[random_index]
    
    def shorten_time_func(self, array):
        video_length = len(array)
        progress = 1
        if len(array) > 34:
            max_len = min(32, len(array) - 2)
            # random choose end from 32, max_len
            end = random.randint(32, max_len)
            array = array[:end]

            progress = end / video_length
            
        return array, progress
    
    def sample_negative_func(self, key):
        other_key = key
        while other_key == key:
            other_key = random.choice(self.keys)
        other_group = self.h5_file[other_key]
        other_gif_names = list(other_group.keys())
        neg_gif_name = random.choice(other_gif_names)
        neg_array = other_group[neg_gif_name][()]
        return neg_array

    def sample_frames(self, array, num_frames = 32):
        if len(array) > num_frames:
            indices = np.linspace(0, len(array) - 1, num_frames, dtype=int)
            return array[indices]
        else:
            more_frames = num_frames - len(array)
            last_frame = array[-1:]
            for i in range(more_frames):
                array = np.concatenate([array, last_frame], axis=0)
        return array

    def preprocess_xclip(self, array):
        # crop to from 250x250 to 224x224
        # if array.shape != (32, 250, 250, 3):

        array = array[:, 13:237, 13:237, :]
        pixel_values = self.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values

    def __del__(self):
        self.h5_file.close()

class GifProgressDataset(GifDataset):
    def __init__(self, args):
        # init parent class first
        super(GifProgressDataset, self).__init__(args)
        self.candidate_type = [1]
        if args.time_shuffle:
            self.candidate_type.append(2)
        if args.time_shorten:
            self.candidate_type.append(3)
        


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        group = self.h5_file[key]
        gif_names = list(group.keys())

        # sample gt sample
        gt_gif_name = random.choice(gif_names)
        gt_array = group[gt_gif_name][()]

        # sample positive sample
        pos_gif_name = random.choice(gif_names)
        pos_array = group[pos_gif_name][()]

        
        # sample negative sample
        # type 1: other key 
        # type 2: shuffle time 
        # type 3: shorten time

        neg_type = random.choice(self.candidate_type)
        progress = 0
        if neg_type == 1:
            neg_array = self.sample_negative_func(key)
        elif neg_type == 2:
            neg_array = self.shuffle_time_func(gt_array.copy())
        elif neg_type == 3:
            neg_array, progress = self.shorten_time_func(gt_array.copy())


        # sample frames
        gt_array = self.sample_frames(gt_array)
        pos_array = self.sample_frames(pos_array)
        neg_array = self.sample_frames(neg_array)

        # preprocess
        if self.vlm_name == "xclip":

            gt_array = self.preprocess_xclip(gt_array)
            pos_array = self.preprocess_xclip(pos_array)
            neg_array = self.preprocess_xclip(neg_array)

        else:
            gt_array = gt_array/255
            pos_array = pos_array/255
            neg_array = neg_array/255

            gt_array = gt_array.transpose(3, 0, 1, 2)
            pos_array = pos_array.transpose(3, 0, 1, 2)
            neg_array = neg_array.transpose(3, 0, 1, 2)

            gt_array = th.from_numpy(gt_array).float()
            pos_array = th.from_numpy(pos_array).float()
            neg_array = th.from_numpy(neg_array).float()

        output_dict = {
            "gt_array": gt_array,
            "pos_array": pos_array,
            "neg_array": neg_array,
            "type": neg_type,
            "progress": progress
        }

        return  output_dict
