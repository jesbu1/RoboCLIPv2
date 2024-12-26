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

# from confusion_matrix import plot_confusion_matrix_pca
# import pca 
# from pca_utils import pca_learner
# import joblib

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

    # if args.pca:
    #     experiment_name += "_pca" 

    if args.subtract_before:
        experiment_name += "_subtract_before"
    else:
        experiment_name += "_subtract_after"

    experiment_name += "_heads_" + str(args.attention_heads)

    if args.sample_neg:
        experiment_name += "_sample_neg"
    if args.reverse_video:
        experiment_name += "_reverse_video"
    if args.normalize_embedding:
        experiment_name += "_norm"
    if args.catagorical_progress:
        experiment_name += "_CatProgress"
    if args.subsample_video:
        experiment_name += "_subsample_video"
    if args.cat_embedding:
        experiment_name += "_cat_embedding"



    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Regression_final_debug",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")

    # if args.pca:
    #     pca_video_model, pca_text_model, transform_model = pca_learner(h5_file)
    #     embedding_dim = pca_video_model.components_.shape[0]
    #     transform_model = transform_model.to(device)
    # else:
    embedding_dim = 1024


    # if args.sample_neg:
    dataset = LivVideoDataset(args, h5_file)
    # else:
    #     dataset = ClipLivVideoDataset(args, h5_file)
    if args.subsample_video:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, collate_fn=video_collate_fn)
    if args.catagorical_progress:
        loss_function = CrossEntropyLoss()
    else:
        loss_function = mse_loss


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

    # if args.pca:
    #     optimizer = torch.optim.Adam(list(self_attention_model.parameters()) + list(transform_model.parameters()), lr=args.lr)
    # else:
    optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)



    for epoch in range(args.epochs):
        self_attention_model.train()
        # if args.pca:
        #     transform_model.train()
        for i, data in enumerate(tqdm(dataloader)):
            video_array = data["video_array"].to(device).float()

            text_array = data["text_array"].to(device).float()
            # if args.pca:
            #     text_array = pca_text_model.transform(text_array.cpu().detach().numpy())
            #     text_array = torch.tensor(text_array).to(device).float()

            #     video_array = video_array.view(-1, 1024) # convert to (batch_size * num_frames, 1024)
            #     video_array = pca_video_model.transform(video_array.cpu().detach().numpy())
            #     video_array = torch.tensor(video_array).to(device).float()
            #     video_array = transform_model(video_array)
            #     # convert back to (batch_size, num_frames, 1024)
            #     video_array = video_array.view(args.batch_size, -1, video_array.shape[1])
            if not args.subsample_video:
                mask = data["mask"].to(device).float()
                progress_output = self_attention_model(video_array, mask, text_array)
            else:
                progress_output = self_attention_model(video_array, None, text_array)
            if args.catagorical_progress:
                progress = data["progress"].to(device).long()
            else:
                progress = data["progress"].to(device).float().unsqueeze(1)
            loss = loss_function(progress_output, progress)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {
                "loss": loss.item(),
            }
            if args.catagorical_progress:
                predict_label = torch.argmax(progress_output, dim=1)
                accuracy = torch.sum(predict_label == progress).item() / args.batch_size
                wandb_log["accuracy"] = accuracy

            wandb.log(wandb_log)

        if epoch % 10 == 9:
            self_attention_model.eval()

            save_path = os.path.join("/scr/jzhang96/roboclip_v2_models_final", experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_dict = {
                "model": self_attention_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args
            }
            torch.save(save_dict, os.path.join(save_path, f"model_{epoch}.pth"))


            if args.catagorical_progress:
                
                plot_progress_class(h5_file, args.model_name, "train", self_attention_model, args)
                plot_progress_class(h5_file, args.model_name, "eval", self_attention_model, args)
                plot_confusion_matrix_pca_class(h5_file = h5_file, 
                                          model_name = args.model_name, 
                                          set = "train", 
                                          self_attention_model = self_attention_model, 
                                          args = args)
                plot_confusion_matrix_pca_class(h5_file = h5_file,
                                                model_name = args.model_name,
                                                set = "eval",
                                                self_attention_model = self_attention_model,
                                                args = args)
            
            else:
                plot_progress(h5_file, args.model_name, "train", self_attention_model, args)
                plot_progress(h5_file, args.model_name, "eval", self_attention_model, args)
                plot_confusion_matrix_pca(h5_file = h5_file, 
                                          model_name = args.model_name, 
                                          set = "train", 
                                          self_attention_model = self_attention_model, 
                                          args = args)
                plot_confusion_matrix_pca(h5_file = h5_file,
                                          model_name = args.model_name,
                                          set = "eval",
                                          self_attention_model = self_attention_model,
                                          args = args)

            self_attention_model.train()

        if epoch % 20 == 19:
            self_attention_model.eval()

            if args.catagorical_progress:
                plot_videos_class(args.model_name, self_attention_model, args)
            else:
                plot_videos(args.model_name, self_attention_model, args)

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
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--subtract_before', action='store_true')
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--sample_neg', action='store_true')
    argparser.add_argument('--enlarge_embedding_space', action='store_true')
    argparser.add_argument('--reverse_video', action='store_true')
    argparser.add_argument('--normalize_embedding', action='store_true')    
    argparser.add_argument('--catagorical_progress', action='store_true')
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--catagorical_progress_bins', type=int, default=5)
    argparser.add_argument('--max_length', type=int, default=32)
    argparser.add_argument('--cat_embedding', action='store_true')
    args = argparser.parse_args()
    main(args)






    