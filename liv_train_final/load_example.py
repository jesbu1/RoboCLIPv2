import torch
from PIL import Image
from dataloader_liv import video_collate_fn, LivVideoDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
import wandb
from tqdm import tqdm
import h5py
# from eval_video_self_attention_pca import plot_progress, plot_progress_corr, plot_videos
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from models import MultiHeadAttentionModel, MultiHeadAttentionSubtraction, MultiHeadAttentionConcatenation
from eval_utils import plot_progress, plot_progress_class, plot_videos, plot_videos_class
from confusion_matrix import plot_confusion_matrix_pca, plot_confusion_matrix_pca_class


state_dict_path = "/scr/jzhang96/roboclip_v2_models_final/RegressionRandom_liv_sample_neg_subtract_after_heads_4_sample_neg_reverse_video_norm/model_9.pth"
state_dict = torch.load(state_dict_path)
args = argparse.Namespace()
args = state_dict['args']
print(args)

embedding_dim = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cat_embedding:
    if args.catagorical_progress:
        if args.sample_neg:
            num_bins = args.catagorical_progress_bins + 1
        else:
            num_bins = args.catagorical_progress_bins
        self_attention_model = MultiHeadAttentionConcatenation(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
    else:
        self_attention_model = MultiHeadAttentionConcatenation(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)
else:
    if args.subtract_before:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
            self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
        else:
            self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)
        
    else:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
            self_attention_model = MultiHeadAttentionModel(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
        else:
            self_attention_model = MultiHeadAttentionModel(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)


self_attention_model.load_state_dict(state_dict['model'])
self_attention_model.eval()

print("Model loaded")