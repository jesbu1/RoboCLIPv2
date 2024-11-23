import torch
from PIL import Image
from dataloader_clipliv_video import video_collate_fn, ClipLivVideoNegDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, TwoLayerMLP, pca_learner, compute_M, SingleLayerMLP, ThreeLayerMLP
import wandb
from tqdm import tqdm
import h5py
from eval_video_self_attention_utils import plot_progress, plot_progress_corr, plot_videos, plot_wrong_progress
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from self_attention_utils import MultiHeadAttention, MultiHeadAttentionSubtraction
from confusion_matrix import plot_confusion_matrix
# import pca 
from pca_utils import pca_learner


os.environ["TOKENIZERS_PARALLELISM"] = "False"








def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "RegressionRandomStartVideoMultiHeadNew_" + args.model_name + "_negclass"

    if args.subtract_before:
        experiment_name += "_subtract_before"
    else:
        experiment_name += "_subtract_after"

    experiment_name += "_heads_" + str(args.attention_heads)


    if args.pca:
        experiment_name += "_pca_" + str(args.pca_var)




    



    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="RegressionRandomStartMultiHeadSelfAttentioncompare",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")

    


    dataset = ClipLivVideoNegDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, collate_fn=video_collate_fn)

    loss_function = mse_loss


    if args.model_name == "clip":
        transform_model = TwoLayerMLP(768).to(device)
        self_attention_model = MultiHeadAttention(768).to(device)

    elif args.model_name == "liv":

        if args.subtract_before:
            self_attention_model = MultiHeadAttentionSubtraction(1024, num_heads = args.attention_heads, dropout = args.dropout).to(device)
            
        else:
            self_attention_model = MultiHeadAttention(1024, num_heads = args.attention_heads, dropout = args.dropout).to(device)

    optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)



    for epoch in range(args.epochs):
        self_attention_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            video_array = data["video_array"].to(device).float()
            
            # video
            mask = data["mask"].to(device).float()
            progress_output = self_attention_model(video_array, mask, text_array)
            progress = data["progress"].to(device).float().unsqueeze(1)

            loss = loss_function(progress_output, progress)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {
                "loss": loss.item(),
            }

            wandb.log(wandb_log)

        
        if epoch % 25 == 24:
            self_attention_model.eval()
            with torch.no_grad():
                plot_confusion_matrix(h5_file, args.model_name, "train", self_attention_model)
                plot_confusion_matrix(h5_file, args.model_name, "eval", self_attention_model)
                corr_train_dict = plot_progress_corr(h5_file, args.model_name, "train", self_attention_model)
                corr_eval_dict = plot_progress_corr(h5_file, args.model_name, "eval", self_attention_model)
                wandb.log(corr_train_dict)
                wandb.log(corr_eval_dict)
            
                plot_progress(h5_file, args.model_name, "train", self_attention_model)
                plot_progress(h5_file, args.model_name, "eval", self_attention_model)
            # plot_wrong_progress(h5_file, args.model_name, "train", self_attention_model)
            # plot_wrong_progress(h5_file, args.model_name, "eval", self_attention_model)
            self_attention_model.train()

        
        if epoch % 25 == 24:
            self_attention_model.eval()
            with torch.no_grad():
                plot_videos(args.model_name, self_attention_model)
            self_attention_model.train()




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_25_for_clip_liv.h5')
    argparser.add_argument('--model_name', type=str, default='liv', choices=['clip', 'liv'])
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=100)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--loss_type', type=str, choices=['triplet', 'mse'], default='mse')
    argparser.add_argument('--margin_range', type=float, default=1.0)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--pca_var', type=float, default=1.0)
    argparser.add_argument('--reverse', action='store_true')
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--subtract_before', action='store_true')
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--dropout', type=float, default=0.1)
    args = argparser.parse_args()
    main(args)






    