import torch
from PIL import Image
from dataloader_clipliv_video import video_collate_fn, ClipLivVideoNegDataset, ClipLivVideoDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, TwoLayerMLP, pca_learner, compute_M, SingleLayerMLP, ThreeLayerMLP
import wandb
from tqdm import tqdm
import h5py
from eval_video_self_attention_pca import plot_progress, plot_progress_corr, plot_videos
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from self_attention_utils import MultiHeadAttention, MultiHeadAttentionSubtraction
from confusion_matrix import plot_confusion_matrix_pca
# import pca 
from pca_utils import pca_learner
import joblib



os.environ["TOKENIZERS_PARALLELISM"] = "False"








def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "RegressionRandom_" + args.model_name 

    if args.sample_neg:
        experiment_name += "_sample_neg"

    if args.pca:
        experiment_name += "_pca" 

    if args.subtract_before:
        experiment_name += "_subtract_before"
    else:
        experiment_name += "_subtract_after"

    experiment_name += "_heads_" + str(args.attention_heads)
    
    if args.random_shuffle:
        experiment_name += "_random_shuffle"






    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="RegressionRandomStartMultiHeadSelfAttentionSameLength1_debug",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")

    if args.pca:
        pca_video_model, pca_text_model, transform_model = pca_learner(h5_file)
        embedding_dim = pca_video_model.components_.shape[0]
        transform_model = transform_model.to(device)
    else:
        embedding_dim = 1024


    if args.sample_neg:
        dataset = ClipLivVideoNegDataset(args, h5_file)
    else:
        dataset = ClipLivVideoDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, collate_fn=video_collate_fn)
    loss_function = mse_loss


    if args.subtract_before:
        self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout).to(device)
        
    else:
        self_attention_model = MultiHeadAttention(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout).to(device)

    if args.pca:
        optimizer = torch.optim.Adam(list(self_attention_model.parameters()) + list(transform_model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)



    for epoch in range(args.epochs):
        self_attention_model.train()
        if args.pca:
            transform_model.train()
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            video_array = data["video_array"].to(device).float()
            # video
            mask = data["mask"].to(device).float()
            if args.pca:
                text_array = pca_text_model.transform(text_array.cpu().detach().numpy())
                text_array = torch.tensor(text_array).to(device).float()

                video_array = video_array.view(-1, 1024) # convert to (batch_size * num_frames, 1024)
                video_array = pca_video_model.transform(video_array.cpu().detach().numpy())
                video_array = torch.tensor(video_array).to(device).float()
                video_array = transform_model(video_array)
                # convert back to (batch_size, num_frames, 1024)
                video_array = video_array.view(args.batch_size, -1, video_array.shape[1])

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
            if args.pca:
                transform_model.eval()
            with torch.no_grad():
                if args.pca:
                    plot_videos(args.model_name, self_attention_model, pca_text_model, pca_video_model, transform_model)
                else:
                    plot_videos(args.model_name, self_attention_model)
            self_attention_model.train()
            if args.pca:
                transform_model.train()

        
        if epoch % 10 == 9:
            self_attention_model.eval()
            if args.pca:
                transform_model.eval()
            with torch.no_grad():

                if args.pca:
                    plot_confusion_matrix_pca(h5_file, args.model_name, "train", self_attention_model, pca_text_model, pca_video_model, transform_model)
                    plot_confusion_matrix_pca(h5_file, args.model_name, "eval", self_attention_model, pca_text_model, pca_video_model, transform_model)
                else:
                    plot_confusion_matrix_pca(h5_file, args.model_name, "train", self_attention_model)
                    plot_confusion_matrix_pca(h5_file, args.model_name, "eval", self_attention_model)

                if args.pca:
                    corr_train_dict = plot_progress_corr(h5_file, args.model_name, "train", self_attention_model, pca_text_model, pca_video_model, transform_model)
                    corr_eval_dict = plot_progress_corr(h5_file, args.model_name, "eval", self_attention_model, pca_text_model, pca_video_model, transform_model)
                else:
                    corr_train_dict = plot_progress_corr(h5_file, args.model_name, "train", self_attention_model)
                    corr_eval_dict = plot_progress_corr(h5_file, args.model_name, "eval", self_attention_model)

                wandb_log = {
                    "corr_train": corr_train_dict,
                    "corr_eval": corr_eval_dict
                }
                wandb.log(wandb_log)

                if args.pca:
                    plot_progress(h5_file, args.model_name, "train", self_attention_model, pca_text_model, pca_video_model, transform_model)
                    plot_progress(h5_file, args.model_name, "eval", self_attention_model, pca_text_model, pca_video_model, transform_model)
                else:
                    plot_progress(h5_file, args.model_name, "train", self_attention_model)
                    plot_progress(h5_file, args.model_name, "eval", self_attention_model)


            self_attention_model.train()
            if args.pca:
                transform_model.train()

        if epoch % 25 == 24:
            save_model_path = "/scr/jzhang96/clip_liv_models"
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            folder_name = experiment_name
            if not os.path.exists(os.path.join(save_model_path, folder_name)):
                os.makedirs(os.path.join(save_model_path, folder_name))
            torch.save(self_attention_model.state_dict(), os.path.join(save_model_path, folder_name, f"model_{epoch}.pt"))
            if args.pca:
                torch.save(transform_model.state_dict(), os.path.join(save_model_path, folder_name, f"transform_{epoch}.pt"))
                # dump pca models
                pca_video_path = os.path.join(save_model_path, folder_name, f"pca_video.pkl")
                pca_text_path = os.path.join(save_model_path, folder_name, f"pca_text.pkl")
                joblib.dump(pca_video_model, pca_video_path)
                joblib.dump(pca_text_model, pca_text_path)
            



        





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
    argparser.add_argument('--reverse', action='store_true')
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--subtract_before', action='store_true')
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--sample_neg', action='store_true')
    argparser.add_argument('--random_shuffle', action='store_true')
    args = argparser.parse_args()
    main(args)






    