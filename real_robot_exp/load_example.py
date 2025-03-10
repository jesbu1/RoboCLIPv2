import torch
import numpy as np
import os

from models import ClassProgressTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_dim = 768
text_dim = 384



model_path = "saved_models/AugTextFixRewindMetaWorld_binary_thrd_0.5_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_20_lr_0.0001_progress_loss_weight_2.0/epoch_15.pth"
model_dict = torch.load(model_path)
args = model_dict['args']


self_attention_model = ClassProgressTransformer(
    args=args,
    video_dim=video_dim,  # Original video embedding dimension
    text_dim=text_dim,   # Original text embedding dimension
    hidden_dim=512  # Common dimension for transformer processing
).to(device)
self_attention_model.load_state_dict(model_dict['model'])


