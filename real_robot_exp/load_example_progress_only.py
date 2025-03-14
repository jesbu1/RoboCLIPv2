from models import ClassProgressTransformer
import os 
import torch
# Load the model
path = os.path.join("saved_models", 
                    "ProgressOnlyCrop_MetaWorld_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_50_lr_0.0001_progress_loss_weight_1",
                    "epoch_35.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_dim = 768
text_dim = 384

model_dict = torch.load(model_path)
args = model_dict['args']
model = ClassProgressTransformer(
        args=args,
        video_dim=video_dim,  # Original video embedding dimension
        text_dim=text_dim,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)

model.load_state_dict(model_dict['model'])
model.eval()

# Load the data



