import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
import math
from tqdm import tqdm
class CosineWithMinLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            # Cosine decay for the first max_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos_decay for _ in self.base_lrs]
        else:
            # Keep the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]




def eval_model(positive_eval_openx_dataset, video_encoder, progress_loss_function, triangular_mask, args, pca_text_model = None, pca_video_model = None, neg_sample = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_num = 0
    total_loss = 0

    # wrong_num = 0
    for eval_data in tqdm(positive_eval_openx_dataset):
        
        video_array = eval_data["video_array"].to(device).float()
        text_array = eval_data["text_array"].to(device).float().squeeze(1)
        progress = eval_data["progress"].to(device)
        progress_mask = torch.ones_like(progress).bool()
        if neg_sample:
            stop_idx = round(video_array.size(1) * args.negative_mask_ratio)
            progress_mask[:, :stop_idx] = False

        batch_triangular_mask = triangular_mask.repeat(video_array.size(0), 1, 1, 1).bool()

        video_embedding = video_encoder(video_array, batch_triangular_mask)
        text_array = text_array.unsqueeze(1).repeat(1, video_embedding.size(1), 1)

        video_embedding = video_embedding.view(video_embedding.size(0) * video_embedding.size(1), -1)
        text_array = text_array.view(text_array.size(0) * text_array.size(1), -1)

        if args.norm_length:
            video_embedding = F.normalize(video_embedding, p=2, dim=1)
            text_array = F.normalize(text_array, p=2, dim=1)
        dot_product = torch.sum(video_embedding * text_array, dim = 1)
        dot_product = dot_product.view(-1, args.max_length)

        pred = dot_product
        target = progress

        loss = progress_loss_function(pred[progress_mask], target[progress_mask])
        total_loss += loss.item()
        total_num += 1
    progress_loss = total_loss / total_num
    
    return progress_loss


