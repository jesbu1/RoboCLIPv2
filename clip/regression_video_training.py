import torch
from PIL import Image
from dataloader_clipliv_video import ClipLivVideoMeanDataset
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
from eval_video_rank_utils import plot_progress, plot_progress_corr, plot_videos
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
    experiment_name = "RegressionRandomStartVideoLossFix_" + args.model_name + "_" + args.loss_type

    if args.pca:
        experiment_name += "_pca_" + str(args.pca_var)
        if args.pca_only_goal:
            experiment_name += "_only_goal"

    if args.loss_type == "triplet":
        experiment_name += "_margin_" + str(args.margin_range)
    
    if args.subtract:
        experiment_name += "_subtract"


    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="RegressionRandomStartVideoLossSubstractionFix",
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


    dataset = ClipLivVideoMeanDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    loss_function = mse_loss


    # if args.pca:
    #     transform_model = TwoLayerMLP(image_pca_model.components_.shape[0] * (3 - args.subtract)).to(device) # start_frame, end_frame, text
    # else:

    if args.model_name == "clip":
        transform_model = TwoLayerMLP(768 * (2 - args.subtract)).to(device)
    elif args.model_name == "liv":
        transform_model = TwoLayerMLP(1024 * (2 - args.subtract)).to(device)


    optimizer = torch.optim.Adam(transform_model.parameters(), lr=args.lr)


    # corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract)
    # corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_videos(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model, args.subtract)

    if args.learn_params:
        if args.model_name == "clip":
            context_parameters = torch.nn.Parameter(torch.randn(1, 768))
            context_parameters.requires_grad = True
        elif args.model_name == "liv":
            context_parameters = torch.nn.Parameter(torch.randn(1, 1024))
            context_parameters.requires_grad = True

        optimizer = torch.optim.Adam(list(transform_model.parameters()) + [context_parameters], lr=args.lr)



    for epoch in range(args.epochs):
        transform_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            # video_array = normalize_embeddings(data["video_array"].to(device)).float()
            video_array = data["video_array"].to(device).float()

            if args.subtract:
                progress_input = video_array - text_array
            else:
                progress_input = torch.cat([text_array, video_array], dim=1)


            progress = data["progress"].to(device).float().unsqueeze(1)
            feature_dim = text_array.shape[1]

            progress_output = transform_model(progress_input)
            loss = loss_function(progress_output, progress)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {
                "loss": loss.item(),
                "feature_dim": feature_dim
            }

            wandb.log(wandb_log)


        if epoch % 5 == 4:
            corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", args.subtract)
            corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", args.subtract)
            wandb.log(corr_train_dict)
            wandb.log(corr_eval_dict)
            
            plot_progress(h5_file, args.model_name, transform_model, "train", args.subtract)
            plot_progress(h5_file, args.model_name, transform_model, "eval", args.subtract)

        
        if epoch % 20 == 19:
            plot_videos(args.model_name, transform_model, args.subtract)




        




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
    argparser.add_argument('--mean_pool', action='store_true')
    argparser.add_argument('--learn_params', action='store_true')
    args = argparser.parse_args()
    main(args)






    