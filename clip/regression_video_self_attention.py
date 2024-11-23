import torch
from PIL import Image
from dataloader_clipliv_video import ClipLivVideoDataset, video_collate_fn, ClipLivVideoReverseDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, TwoLayerMLP, pca_learner, compute_M, SingleLayerMLP, ThreeLayerMLP
from torch.nn import MarginRankingLoss
import wandb
from tqdm import tqdm
import h5py
from eval_video_self_attention_utils import plot_progress, plot_progress_corr, plot_videos
from torch.nn.functional import mse_loss
import os
from self_attention_utils import SingleHeadAttentionWithPositionalEncoding, SingleHeadAttentionWithOutPositionalEncoding

os.environ["TOKENIZERS_PARALLELISM"] = "False"








def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "RegressionRandomStartVideoLoss_" + args.model_name + "_" + args.loss_type + "_" + str(args.layers) + "layers"
    if args.reverse:
        experiment_name += "_reverse"
    experiment_name += "last"
    if args.position_encoding:
        experiment_name += "_pos_encoding"

    if args.pca:
        experiment_name += "_pca_" + str(args.pca_var)
        if args.pca_only_goal:
            experiment_name += "_only_goal"

    if args.loss_type == "triplet":
        experiment_name += "_margin_" + str(args.margin_range)


    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="RegressionRandomStartVideoDiag",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")

    if args.reverse:
        dataset = ClipLivVideoReverseDataset(args, h5_file)
    else:
        dataset = ClipLivVideoDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, collate_fn=video_collate_fn)

    loss_function = mse_loss


    if args.model_name == "clip":
        transform_model = TwoLayerMLP(768).to(device)
        if args.position_encoding:
            self_attention_model = SingleHeadAttentionWithPositionalEncoding(768).to(device)
        else:
            self_attention_model = SingleHeadAttentionWithOutPositionalEncoding(768).to(device)
        # self_attention_model = SingleHeadAttentionWithPositionalEncoding(768).to(device)
    elif args.model_name == "liv":
        if args.layers == 2:
            transform_model = TwoLayerMLP(1024).to(device)
        elif args.layers == 3:
            transform_model = ThreeLayerMLP(1024).to(device)
        if args.position_encoding:
            self_attention_model = SingleHeadAttentionWithPositionalEncoding(1024).to(device)
        else:
            self_attention_model = SingleHeadAttentionWithOutPositionalEncoding(1024).to(device)

    # transformer model and self_attention model
    optimizer = torch.optim.Adam(list(transform_model.parameters()) + list(self_attention_model.parameters()), lr=args.lr)



    for epoch in range(args.epochs):
        transform_model.train()
        self_attention_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            # video_array = normalize_embeddings(data["video_array"].to(device)).float()
            video_array = data["video_array"].to(device).float()
            
            # video
            mask = data["mask"].to(device).float()
            video_embedding = self_attention_model(video_array, mask)


            progress_input = video_embedding - text_array
            progress = data["progress"].to(device).float().unsqueeze(1)
            feature_dim = text_array.shape[1]
            progress_output = transform_model(progress_input)
            loss = loss_function(progress_output, progress)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {
                "loss": loss.item(),
                "feature_dim": feature_dim,
            }

            wandb.log(wandb_log)


        if epoch % 5 == 4:
            with torch.no_grad():
                corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", self_attention_model)
                corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", self_attention_model)
                wandb.log(corr_train_dict)
                wandb.log(corr_eval_dict)
            
            plot_progress(h5_file, args.model_name, transform_model, "train", self_attention_model)
            plot_progress(h5_file, args.model_name, transform_model, "eval", self_attention_model)

        
        # if epoch % 50 == 49:
        #     plot_videos(args.model_name, transform_model, self_attention_model)




        




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
    argparser.add_argument('--pca_only_goal', action='store_true')
    argparser.add_argument('--pca_var', type=float, default=1.0)
    argparser.add_argument('--reverse', action='store_true')
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--position_encoding', action='store_true')
    argparser.add_argument('--layers', type=int, default=2)
    args = argparser.parse_args()
    main(args)






    