import random
import torch as th
import torch.nn as nn
# help me import attrdict
import kornia.augmentation as K
import torch.nn.functional as F
import numpy as np
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


def triplet_loss(gt, positive, negative, type, margin = (1.0, 1.0, 1.0, 0.0, 1.0), progress = None): 

    pos_sim = cosine_similarity(gt, positive)
    neg_sim = cosine_similarity(gt, negative)

    loss = torch.zeros_like(pos_sim).cuda()

    # type1: hard negative margin = 1.5
    mask_type_1 = (type == 1)
    loss[mask_type_1] = F.relu(margin[0] - pos_sim[mask_type_1] + neg_sim[mask_type_1])

    # type2: semi-hard negative margin = 1.2
    mask_type_2 = (type == 2)
    loss[mask_type_2] = F.relu(margin[1] - pos_sim[mask_type_2] + neg_sim[mask_type_2])

    # type3: adaptive margin
    mask_type_3 = (type == 3)
    progress = progress[mask_type_3]
    # adaptive margin range from 1.0 to 0.0
    adaptive_margin = (margin[2] + (margin[3] - margin[2]) * progress).cuda()
    loss[mask_type_3] = F.relu(adaptive_margin - pos_sim[mask_type_3] + neg_sim[mask_type_3])

    mask_type_4 = (type == 4)
    loss[mask_type_4] = F.relu(margin[3] - pos_sim[mask_type_4] + neg_sim[mask_type_4])

    return loss.mean()


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)




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
                erase_p=0.00,
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
                erase_p=0.0,
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
                erase_p=0.0,
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
        batch_size, num_frames, _, _, _ = x.shape  # (batch_size, 32, 3, 224, 224)
        x = x.to(self.device)
        
        # Prepare first frames for generating augmentation parameters
        first_frames = x[:, 0, :, :, :]  # Shape: (batch_size, 3, 224, 224)

        # Generate augmentation parameters for each video in the batch
        video_params = []
        for name, aug in self.augmentations.named_children():
            # Set random seeds for reproducibility
            seeds = th.randint(0, 100000, (batch_size,))
            th.manual_seed(seeds[0].item())
            random.seed(seeds[0].item())

            # Get parameters for all videos in batch
            params = aug.forward_parameters(first_frames.shape)

            # Adjust parameters for each video in the batch
            video_params.append({name: {
                k: v.repeat_interleave(num_frames, dim=0) if k not in ['order'] else v
                for k, v in params.items()
            }})

        # Apply augmentations to the entire batch with the generated parameters
        x = x.view(batch_size * num_frames, *x.shape[2:])  # Reshape to (batch_size * 32, 3, 224, 224)
        for aug_params in video_params:
            for name, params in aug_params.items():
                x = self.augmentations[int(name)](x, params=params)

        x = x.view(batch_size, num_frames, *x.shape[1:])  # Reshape back to (batch_size, 32, 3, 224, 224)
        return x



    # def forward(self, x):
    #     # 将视频张量从 (1, 3, 32, 224, 224) 转换为 (32, 3, 224, 224)
    #     x = x.squeeze(0).permute(1, 0, 2, 3)  # (32, 3, 224, 224)
    #     x = x.to(self.device)
    #     # 使用视频的第一帧生成增强参数
    #     first_frame = x[0].unsqueeze(0)  # (1, 3, 224, 224)
    #     video_params = {}
    #     for name, aug in self.augmentations.named_children():
    #         # 为每个增强操作设置不同的随机种子
    #         seed = random.randint(0, 100000)
    #         th.manual_seed(seed)
    #         random.seed(seed)

    #         # 获取增强参数
    #         params = aug.forward_parameters(first_frame.shape)
    #         # 根据参数的性质,决定是否复制32次
    #         video_params[name] = {}
    #         for k, v in params.items():
    #             if k in ['order']:
    #                 # 这些参数在所有帧中保持不变
    #                 video_params[name][k] = v
    #             else:
    #                 # 其他参数复制32次,以匹配视频帧数
    #                 video_params[name][k] = v.repeat(x.shape[0], *[1] * (v.dim() - 1))

    #         # 修改 forward_input_shape 以匹配整个视频张量的形状
    #         video_params[name]['forward_input_shape'] = th.tensor(x.shape)
    #     #print(video_params)
    #         # 对整个视频张量应用相同的增强参数
    #     for name, params in video_params.items():
    #         x = self.augmentations[int(name)](x, params=params)

    #         # 转换回 (1, 3, 32, 224, 224)
    #     x = x.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 32, 224, 224)
    #     # return x.cpu()
    #     return x

