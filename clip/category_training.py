import torch
from PIL import Image
from dataloader_clipliv import ClipLivDataset, ClipLivProgressDataset, ClipLivSingleDataset, ClipLivCategoryDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, TwoLayerMLP, pca_learner, compute_M, SingleLayerMLP, TwoLayerClassMLP
from torch.nn import MarginRankingLoss
import wandb
from tqdm import tqdm
import h5py
from eval_rank_utils import plot_class_progress, plot_videos_class
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss
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
    experiment_name = "CategoroalLoss_" + args.model_name 
    if args.sample_neg:
        experiment_name += "_sample_neg"
    experiment_name += "_num_classes_" + str(args.num_classes)

    if args.pca:
        experiment_name += "_pca_" + str(args.pca_var)
        if args.pca_only_goal:
            experiment_name += "_only_goal"

    if args.subtract:
        experiment_name += "_subtract"


    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Categoroal_loss_train_substraction4",
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



    dataset = ClipLivCategoryDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3, drop_last=True)
    num_classes = args.num_classes
    if args.sample_neg:
        num_classes = args.num_classes + 1

    loss_function = CrossEntropyLoss()
    if args.pca:
        transform_model = TwoLayerClassMLP(image_pca_model.components_.shape[0] * (2 - args.subtract), num_classes).to(device)
    else:
        if args.model_name == "clip":
            transform_model = TwoLayerClassMLP(768 * (2 - args.subtract), num_classes).to(device)
        elif args.model_name == "liv":
            transform_model = TwoLayerClassMLP(1024 * (2 - args.subtract), num_classes).to(device)


    optimizer = torch.optim.Adam(transform_model.parameters(), lr=args.lr)


    # corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model)
    # corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model)
    # plot_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model)
    # plot_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model)
    # plot_videos(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model)

    for epoch in range(args.epochs):
        transform_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            progress_array = normalize_embeddings(data["video_array"].to(device)).float()
            progress_class = data["progress_class"].to(device)

            if args.pca:
                
                text_array = text_pca_model.transform(text_array.detach().cpu().numpy())
                text_array = torch.tensor(text_array).to(device).float()
                with torch.no_grad():
                    progress_array = image_pca_model.transform(progress_array.detach().cpu().numpy())
                    progress_array = torch.tensor(progress_array).to(device).float()
                    progress_array = linear_model(progress_array)


            feature_dim = text_array.shape[1]


            # process progress feature
            if args.subtract:
                progress_input = progress_array - text_array
            else:
                progress_input = torch.cat([text_array, progress_array], dim=1)
            pred_class = transform_model(progress_input)

            loss = loss_function(pred_class, progress_class)

            with torch.no_grad():
                pred_class = torch.argmax(pred_class, dim=1)
                
                accuracy = torch.sum(pred_class == progress_class).item()
                accuracy = accuracy / args.batch_size

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {
                "loss": loss.item(),
                "feature_dim": feature_dim,
                "pred_accuracy": accuracy
            }


            wandb.log(wandb_log)


        if epoch % 5 == 4:
            plot_class_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model, num_classes, args.subtract)
            plot_class_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model, num_classes, args.subtract)
        if epoch % 20 == 19:
            plot_videos_class(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model, num_classes, args.subtract)

            # save model
            save_dict = {
                "transform_model": transform_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.pca:
                save_dict["linear_model"] = linear_model.state_dict()


            save_path = "save_models"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path = os.path.join(save_path, experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save(save_dict, f"{save_path}/model_{epoch}.pth")
        #     corr_train_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model)
        #     corr_eval_dict = plot_progress_corr(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model)
        #     plot_progress(h5_file, args.model_name, transform_model, "train", text_pca_model, image_pca_model, linear_model)
        #     plot_progress(h5_file, args.model_name, transform_model, "eval", text_pca_model, image_pca_model, linear_model)
        #     plot_videos(args.model_name, transform_model, text_pca_model, image_pca_model, linear_model)

        #     wandb_progress_log = {}
        #     wandb_progress_log.update(corr_train_dict)
        #     wandb_progress_log.update(corr_eval_dict)
        #     wandb.log(wandb_progress_log)

        




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_25_for_clip_liv.h5')
    argparser.add_argument('--model_name', type=str, default='clip')
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=100)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--margin_range', type=float, default=1.0)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--pca_only_goal', action='store_true')
    argparser.add_argument('--pca_var', type=float, default=1.0)
    argparser.add_argument('--num_classes', type=int, default=5)
    argparser.add_argument('--sample_neg', action='store_true')
    argparser.add_argument('--subtract', action='store_true')

    args = argparser.parse_args()
    main(args)






    