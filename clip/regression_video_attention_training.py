import torch
from PIL import Image
from dataloader_clipliv_video import ClipLivVideoDataset, video_collate_fn, ClipLivVideoReverseDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, TwoLayerMLP, pca_learner, compute_M, SingleLayerMLP
from torch.nn import MarginRankingLoss
import wandb
from tqdm import tqdm
import h5py
from eval_video_attention_utils import plot_progress, plot_progress_corr, plot_videos
from torch.nn.functional import mse_loss
import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"








def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "RegressionRandomStartVideoLoss_" + args.model_name + "_" + args.loss_type

    if args.pca:
        experiment_name += "_pca_" + str(args.pca_var)
        if args.pca_only_goal:
            experiment_name += "_only_goal"

    if args.loss_type == "triplet":
        experiment_name += "_margin_" + str(args.margin_range)
    
    if args.subtract:
        experiment_name += "_subtract1"
    
    if args.reverse:
        experiment_name += "_reverse"


    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="RegressionRandomStartVideoDiag",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")
    # text_pca_model, image_pca_model, linear_model = None, None, None
    # if args.pca:
    #     text_pca_model, image_pca_model = pca_learner(h5_file, args.model_name, args.pca_only_goal, args.pca_var, experiment_name)
    #     computed_matrix = compute_M(image_pca_model.components_, text_pca_model.components_)
    #     linear_model = SingleLayerMLP(image_pca_model.components_.shape[0], text_pca_model.components_.shape[0]).to(device)
    #     linear_model.linear.weight.data = computed_matrix.to(device).float()
    #     linear_model.linear.requires_grad = False

    if args.reverse:
        dataset = ClipLivVideoReverseDataset(args, h5_file)
    else:
        dataset = ClipLivVideoDataset(args, h5_file)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=video_collate_fn)

    loss_function = mse_loss


    # if args.pca:
    #     transform_model = TwoLayerMLP(image_pca_model.components_.shape[0] * (3 - args.subtract)).to(device) # start_frame, end_frame, text
    # else:

    if args.model_name == "clip":
        transform_model = TwoLayerMLP(768 * (2 - args.subtract)).to(device)
    elif args.model_name == "liv":
        transform_model = TwoLayerMLP(1024 * (2 - args.subtract)).to(device)


    


    # corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract)
    # corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_videos(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model, args.subtract)

    if args.learn_params:
        if args.model_name == "clip":
            context_parameters = torch.nn.Parameter(torch.randn(1, 1, 768, device=device), requires_grad=True)
            # context_parameters.requires_grad = True
            # context_parameters = context_parameters.to(device)
        elif args.model_name == "liv":
            context_parameters = torch.nn.Parameter(torch.randn(1, 1, 1024, device=device), requires_grad=True)
            # context_parameters.requires_grad = True
            # context_parameters = context_parameters.to(device)

    if args.learn_params:
        optimizer = torch.optim.Adam(transform_model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(transform_model.parameters()) + [context_parameters], lr=args.lr)



    for epoch in range(args.epochs):
        transform_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            # video_array = normalize_embeddings(data["video_array"].to(device)).float()
            video_array = data["video_array"].to(device).float()
            
            # video
            mask = data["mask"].to(device).float()

            dot_product = (video_array * context_parameters).sum(dim=-1) 
            masked_dot_product = dot_product.masked_fill(mask == 0, -1e9)
            weights = F.softmax(masked_dot_product, dim=1) # batch_size x num_frames
            weights_expanded = weights.unsqueeze(-1) # batch_size x 1 x num_frames
            weighted_video_data = video_array * weights_expanded 
            weighted_video = weighted_video_data.sum(dim=1) 


            if args.subtract:
                progress_input = weighted_video - text_array
            else:
                progress_input = torch.cat([text_array, weighted_video], dim=1)


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
                "avg_weight": weights.mean().item(),
                "max_weight": weights.max().item(),
                "min_weight": weights.min().item(),
                "max_progress_output": progress_output.max().item(),
                "min_progress_output": progress_output.min().item(),
                "avg_progress_output": progress_output.mean().item(),
                "max_progress": progress.max().item(),
                "min_progress": progress.min().item(),
                "avg_progress": progress.mean().item()

            }

            wandb.log(wandb_log)


        if epoch % 5 == 4:
            with torch.no_grad():
                corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", args.subtract, context_parameters)
                corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", args.subtract, context_parameters)
                wandb.log(corr_train_dict)
                wandb.log(corr_eval_dict)
            
            plot_progress(h5_file, args.model_name, transform_model, "train", args.subtract, context_parameters)
            plot_progress(h5_file, args.model_name, transform_model, "eval", args.subtract, context_parameters)

        
        # if epoch % 20 == 19:
        #     plot_videos(args.model_name, transform_model, args.subtract, context_parameters)




        




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
    argparser.add_argument('--subtract', action='store_true')
    argparser.add_argument('--pca_var', type=float, default=1.0)
    argparser.add_argument('--learn_params', action='store_true')
    argparser.add_argument('--reverse', action='store_true')
    args = argparser.parse_args()
    main(args)






    