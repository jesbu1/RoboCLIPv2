# dataset and dataloader
from torch.utils.data import Dataset, DataLoader
import h5py
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from s3dg import S3D
import torch as th
import random
import numpy as np
import copy
import torch.nn as nn
import kornia.augmentation as K

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AugmentationPipeline(nn.Module):
    def __init__(self, device, strength='normal'):
        super(AugmentationPipeline, self).__init__()
        self.device = device

        self.configs = {
            'weak': AttrDict(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                color_p=0.5,
                noise_std=0.05,
                noise_p=0.1,
                channel_shuffle_p=0.0,
                degrees=5,
                translate=0.05,
                affine_p=0.3,
                erase_p=0.05,
            ),
            'normal': AttrDict(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.3,
                color_p=0.9,
                noise_std=0.1,
                noise_p=0.0,
                channel_shuffle_p=0.0,
                degrees=15,
                translate=0.1,
                affine_p=0.6,
                erase_p=0.1,
            ),
            'strong': AttrDict(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
                color_p=0.9,
                noise_std=0.2,
                noise_p=0.2,
                channel_shuffle_p=0.2,
                degrees=25,
                translate=0.2,
                affine_p=0.8,
                erase_p=0.25,
            )
        }

        cfg = self.configs.get(strength, self.configs['normal'])

        self.augmentations = nn.Sequential(
            K.ColorJitter(
                cfg.brightness,
                cfg.contrast,
                cfg.saturation,
                cfg.hue,
                p=cfg.color_p,
            ),
            K.RandomGaussianNoise(std=cfg.noise_std, p=cfg.noise_p),
            K.RandomChannelShuffle(p=cfg.channel_shuffle_p),
            K.RandomAffine(
                degrees=cfg.degrees,
                translate=(cfg.translate, cfg.translate),
                p=cfg.affine_p,
            ),
            K.RandomErasing(p=cfg.erase_p),
        ).to(self.device)

    def forward(self, x):
        # 将视频张量从 (1, 3, 32, 224, 224) 转换为 (32, 3, 224, 224)
        x = x.squeeze(0).permute(1, 0, 2, 3)  # (32, 3, 224, 224)
        x = x.to(self.device)
        # 使用视频的第一帧生成增强参数
        first_frame = x[0].unsqueeze(0)  # (1, 3, 224, 224)
        video_params = {}
        for name, aug in self.augmentations.named_children():
            # 为每个增强操作设置不同的随机种子
            seed = random.randint(0, 100000)
            th.manual_seed(seed)
            random.seed(seed)

            # 获取增强参数
            params = aug.forward_parameters(first_frame.shape)
            # 根据参数的性质,决定是否复制32次
            video_params[name] = {}
            for k, v in params.items():
                if k in ['order']:
                    # 这些参数在所有帧中保持不变
                    video_params[name][k] = v
                else:
                    # 其他参数复制32次,以匹配视频帧数
                    video_params[name][k] = v.repeat(x.shape[0], *[1] * (v.dim() - 1))

            # 修改 forward_input_shape 以匹配整个视频张量的形状
            video_params[name]['forward_input_shape'] = th.tensor(x.shape)
        #print(video_params)
            # 对整个视频张量应用相同的增强参数
        for name, params in video_params.items():
            x = self.augmentations[int(name)](x, params=params)

            # 转换回 (1, 3, 32, 224, 224)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 32, 224, 224)
        return x.cpu()


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


class GifTextDataset(Dataset):
    def __init__(self, args, augmented=False, augmentation_transform=AugmentationPipeline):
        self.h5_file = h5py.File(args.h5_path, "r")
        self.h5_text_file = h5py.File("metaworld_s3d_text.h5", "r")
        self.keys = list(self.h5_file.keys())
        # self.keys = ['door-close-v2-goal-hidden']

        self.augmentation_transform = augmentation_transform
        self.augmented = augmented

        self.vlm_name = args.model_name
        # if self.vlm_name == "xclip":
            # self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
            # self.model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").cuda()
        # else:
        # self.s3d_model = load_model("s3d")  
        self.shuffle_time = args.time_shuffle
        self.shorten_time = args.time_shorten
        self.random_sample_neg = args.rand_neg

        self.candidate_type = [1]
        if args.time_shuffle:
            self.candidate_type.append(2)
        if args.time_shorten:
            self.candidate_type.append(3)

    def __len__(self):
        return len(self.keys) * 50
        # return len(self.keys)


    def __getitem__(self, idx):
        real_idx = idx % len(self.keys) # env name
        key = self.keys[real_idx]
        group = self.h5_file[key]
        gif_names = list(group.keys())
        # sample gt sample
        gt_array = np.asarray(self.h5_text_file[key]["embedding"])

        # sample positive sample
        pos_gif_name = random.choice(gif_names)
        pos_array = group[pos_gif_name][()]

        neg_type = random.choice(self.candidate_type)

        progress = 0

        if neg_type == 2:
            # time shuffle
            neg_array = self.shuffle_time_func(pos_array.copy())
        elif neg_type == 3:
            # time shorten
            neg_array, progress = self.shorten_time_func(pos_array.copy())
        elif neg_type == 1:
            if self.random_sample_neg:
                neg_array = self.sample_negative_func(key)
            else:
                neg_array = copy.deepcopy(pos_array)

        # sample frames
        # gt_array = self.sample_frames(gt_array)
        pos_array = self.sample_frames(pos_array)
        neg_array = self.sample_frames(neg_array)

        # preprocess
        if self.vlm_name == "xclip":

            # gt_array = self.preprocess_xclip(gt_array)
            pos_array = self.preprocess_xclip(pos_array)
            neg_array = self.preprocess_xclip(neg_array)

        else:
            # gt_array = gt_array/255
            pos_array = pos_array/255
            neg_array = neg_array/255

            # gt_array = gt_array.transpose(3, 0, 1, 2)
            pos_array = pos_array.transpose(3, 0, 1, 2)
            neg_array = neg_array.transpose(3, 0, 1, 2)

            gt_array = th.from_numpy(gt_array).float()
            pos_array = th.from_numpy(pos_array).float()
            neg_array = th.from_numpy(neg_array).float()

        if self.augmented:
            pos_array = self.augmentation_transform(pos_array)
            neg_array = self.augmentation_transform(neg_array)

        output_dict = {
            "gt_array": gt_array,
            "pos_array": pos_array,
            "neg_array": neg_array,
            "type": neg_type,
            "progress": progress,
        }

        return output_dict


    def shuffle_time_func(self, array):
        random_index = np.random.permutation(len(array))
        return array[random_index]
    
    def shorten_time_func(self, array):
        video_length = len(array)
        progress = 1
        if len(array) > 33:
            max_len = min(32, len(array) - 1)
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
