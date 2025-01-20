import torch
from PIL import Image
from dataloader_liv_decoder import video_collate_triangular_fn, LivVideoDecoderDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
import wandb
from tqdm import tqdm
import h5py
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from decoder_only_models import RewardPredictor, RewardTwoStepPredictor
from eval_utils_decoder import plot_progress_class, plot_videos_class
from confusion_matrix_decoder import plot_confusion_matrix_pca_class_pdf

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    model_base_path = "/scr/jzhang96/roboclip_v2_decoder_models_fix_2nd"
    model_name = f"RegressionFixTransformer_liv_sample_neg_heads_4_SampleNeg_ReverseVideo_norm"
    # model_name = f"RegressionFixTransformer_liv_sample_neg_heads_4_SampleNeg_ReverseVideo_norm_CatProgress"
    eval_epoch = 459
    epoch_name = f"model_{eval_epoch}.pth"
    model_path = os.path.join(model_base_path, model_name, epoch_name)
    
    state_dict = torch.load(model_path)
    args = argparse.Namespace()
    args = state_dict['args']
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    wandb.init(
        project=WANDB_PROJECT_NAME, 
        entity=WANDB_ENTITY_NAME, 
        group="OfflineEvalDecoder",
        config=args,
        name=model_name + "_" + epoch_name + "_confusion_matrix"
        )




    if args.two_step_training:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
        else:
            num_bins = 1
        self_attention_model = RewardTwoStepPredictor(1024, args = args, class_num=num_bins).to(device)
    else:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
        else:
            num_bins = 1
        self_attention_model = RewardPredictor(1024, args = args, class_num=num_bins).to(device)

    self_attention_model.load_state_dict(state_dict['model'])
    self_attention_model.eval()


    h5_file = h5py.File("metaworld_embedding_1_demo_dataset1.h5", 'r')
    plot_confusion_matrix_pca_class_pdf(h5_file, args.model_name, "train", self_attention_model, args)
    plot_confusion_matrix_pca_class_pdf(h5_file, args.model_name, "eval", self_attention_model, args)

    # plot_videos_class(args.model_name, self_attention_model, args)

if __name__ == "__main__":
    main()

