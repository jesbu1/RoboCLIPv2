import torch
from PIL import Image
from dataloader_clipliv import ClipLivRandomStartDataset
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
from eval_rank_utils import plot_progress, plot_progress_corr, plot_videos
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
    experiment_name = "RegressionRandomStartLoss_" + args.model_name + "_" + args.loss_type

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
        group="RegressionRandomStartLossSubstraction",
        config=args,
        name=experiment_name,
    )




    h5_file = h5py.File(args.h5_embedding_path, "r")
    text_pca_model, image_pca_model, linear_model = None, None, None
    if args.pca:
        text_pca_model, image_pca_model = pca_learner(h5_file, args.model_name, args.pca_only_goal, args.pca_var, experiment_name)
        computed_matrix = compute_M(image_pca_model.components_, text_pca_model.components_)
        linear_model = SingleLayerMLP(image_pca_model.components_.shape[0], text_pca_model.components_.shape[0]).to(device)
        linear_model.linear.weight.data = computed_matrix.to(device).float()
        linear_model.linear.requires_grad = False





    dataset = ClipLivRandomStartDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # if args.loss_type == "triplet":
    #     loss_function = AdaptiveMarginTripletLoss
    # elif args.loss_type == "mse":
    loss_function = mse_loss
    # else:
    #     raise ValueError("Invalid loss type")

    if args.pca:
        transform_model = TwoLayerMLP(image_pca_model.components_.shape[0] * (3 - args.subtract)).to(device) # start_frame, end_frame, text
    else:
        if args.model_name == "clip":
            transform_model = TwoLayerMLP(768 * (3 - args.subtract)).to(device)
        elif args.model_name == "liv":
            transform_model = TwoLayerMLP(1024 * (3 - args.subtract)).to(device)


    optimizer = torch.optim.Adam(transform_model.parameters(), lr=args.lr)


    # corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract)
    # corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract)
    # plot_videos(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model, args.subtract)

    for epoch in range(args.epochs):
        transform_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            start_frame = normalize_embeddings(data["start_frame"].to(device)).float()
            choose_frame = normalize_embeddings(data["choose_frame"].to(device)).float()
            
            progress = data["progress"].to(device).float().unsqueeze(1)

            if args.pca:
                
                text_array = text_pca_model.transform(text_array.detach().cpu().numpy())
                text_array = torch.tensor(text_array).to(device).float()
                with torch.no_grad():
                    start_frame = image_pca_model.transform(start_frame.detach().cpu().numpy())
                    start_frame = torch.tensor(start_frame).to(device).float()
                    start_frame = linear_model(start_frame)

                    choose_frame = image_pca_model.transform(choose_frame.detach().cpu().numpy())
                    choose_frame = torch.tensor(choose_frame).to(device).float()
                    choose_frame = linear_model(choose_frame)


            feature_dim = text_array.shape[1]

        
            # process progress feature
            if args.subtract:
                start_frame = start_frame - text_array
                choose_frame = choose_frame - text_array
                progress_input = torch.cat([start_frame, choose_frame], dim=1)
            else:
                progress_input = torch.cat([text_array, start_frame, choose_frame], dim=1)
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
            corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract, True)
            corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract, True)
            
            plot_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, args.subtract, True)
            plot_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, args.subtract, True)
            wandb.log(corr_train_dict)
            wandb.log(corr_eval_dict)
        
        if epoch % 20 == 19:
            plot_videos(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model, args.subtract, True)

        #     wandb_progress_log = {}
        #     wandb_progress_log.update(corr_train_dict)
        #     wandb_progress_log.update(corr_eval_dict)
        #     wandb.log(wandb_progress_log)

        #     # save model
        #     save_dict = {
        #         "transform_model": transform_model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     if args.pca:
        #         save_dict["linear_model"] = linear_model.state_dict()

        #     save_path = "save_models"
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        #     save_path = os.path.join(save_path, experiment_name)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        #     torch.save(save_dict, f"{save_path}/model_{epoch}.pth")


        




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
    args = argparser.parse_args()
    main(args)






    