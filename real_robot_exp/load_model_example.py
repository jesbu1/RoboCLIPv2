import torch
import torch.nn.functional as F
import numpy as np
import argparse
import wandb
from tqdm import tqdm
import h5py
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss, BCELoss
import os
from models import RewardTwoStepNewPositionEmbeddingPredictor, RewardOneStepNewPositionEmbeddingPredictor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import math
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "False"





model_path = "/home/jzhang96/roboclip_v2_models/MetaWorld_heads_4_CatProgress_SubVideo_MaxLen16_PosEmb_OneStep_FirstFrameEmb_LearnerPara_CosScheduler_ClipGrad_DecoderNum_1_epochs_10000_lr_0.0001/model_900.pth"
state_dict = torch.load(model_path)
args = state_dict["args"]


print(args)



if args.catagorical_progress :
    if not args.two_step_training:
        num_bins = args.catagorical_progress_bins + 1
    else:
        num_bins = args.catagorical_progress_bins
else:
    num_bins = 1

embedding_dim = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.two_step_training:
    self_attention_model = RewardTwoStepNewPositionEmbeddingPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
else:
    self_attention_model = RewardOneStepNewPositionEmbeddingPredictor(embedding_dim, args = args, class_num=num_bins).to(device)


self_attention_model.load_state_dict(state_dict["model"])


text_embedding = torch.rand(size=(1, 1024)).to(device)
video_embedding = torch.rand(size=(1, args.max_length, 1024)).to(device) 
mask = None
triangle_mask =  torch.tril(torch.ones(args.max_length + 1, args.max_length + 1)).to(device).unsqueeze(0).unsqueeze(0)

progress, two_step_class = self_attention_model(video_embedding, triangle_mask, text_embedding, mask)

