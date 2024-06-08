import os
import h5py 
import torch
import argparse
import numpy as np
from models import MILNCELoss
from xclip_utils import DroidH5LatentDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from pca_utils import plot_embeddings
from measure_utils import check_pairs
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def log_wandb(log_dict, data_dict, prefix):
    for key, value in data_dict.items():
        log_dict[prefix + key] = value
    return log_dict



def EvaluateLinearTransformation(text_latent, video_latent):

    # compute the similarity score between the video and text latent features with torch
    text_latent = text_latent.cpu().detach().numpy()
    video_latent = video_latent.cpu().detach().numpy()    

    # sim_score = video_latent @ text_latent.T
    # mean_sim_score = np.mean(sim_score)
    # compute cosine similarity
    mean_sim_score = np.mean(np.diag(video_latent @ text_latent.T))
    accuracies, mrr_k = check_pairs(video_latent, text_latent, None, small_scale=False)

    return mean_sim_score, accuracies, mrr_k

def main(args):
    
    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    WANDB_ENTITY_NAME = "henry_32"
    WANDB_PROJECT_NAME = "roboclip-v2"

    if args.wandb:
        wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group=args.run_group,
            config=args,
            name=args.experiment_name,
        )
    
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Linear(512, 512, bias=args.bias).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.loss == "cosine":
        criterion = torch.nn.CosineSimilarity(dim=1)
    elif args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "nce":
        criterion = MILNCELoss()
    else:
        raise ValueError("Invalid loss function.")

    video_text_dataset = DroidH5LatentDataset(
        h5_path = "/scr/jzhang96/droid_sth_dataset_latent.hdf5",
        debug = False,
    )

    seen_labels = set()
    unique_indices = []
    for idx in tqdm(range(len(video_text_dataset))):
        item = video_text_dataset[idx]
        text_label = item['text']
        if text_label not in seen_labels:
            seen_labels.add(text_label)
            unique_indices.append(idx)
    # video_text_dataset.process_frame = True
    unique_dataset = Subset(video_text_dataset, unique_indices)
    train_dataset, val_dataset = train_test_split(unique_dataset, test_size=0.2, random_state=args.seed)

    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    for i in range (args.num_epochs):
        model.train()
        log = {}
        for batch in tqdm(training_loader):
            video_latent = batch["video_latent"].to(device)
            text_latent = batch["text_latent"].to(device)

            model.zero_grad()
            output = model(video_latent)
            loss = criterion(output, text_latent)
            loss.backward()
            optimizer.step()

            train_mean_sim_score, train_accuracies, train_mrr_k = EvaluateLinearTransformation(text_latent, output)
            log["train_loss"] = loss.item()
            log_dict = log_wandb(log, train_accuracies, "train_")
            log_dict = log_wandb(log_dict, train_mrr_k, "train_")
            log_dict["train_mean_sim_score"] = train_mean_sim_score
        if i == args.num_epochs - 1:
            if args.bias:
                string = "_bias"
            else:
                string = "_no_bias"
            plot_embeddings(output, text_latent, None, directory_name = "train_linear_" + args.loss+string, small_scale=False)
            

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader):
                video_latent = batch["video_latent"].to(device)
                text_latent = batch["text_latent"].to(device)
                output = model(video_latent)
                val_mean_sim_score, val_accuracies, val_mrr_k = EvaluateLinearTransformation(text_latent, output)
                val_loss = criterion(output, text_latent)
                log["val_mean_sim_score"] = val_mean_sim_score
                log_dict = log_wandb(log, val_accuracies, "val_")
                log_dict = log_wandb(log_dict, val_mrr_k, "val_")
                log["val_loss"] = val_loss.item()
        if i == args.num_epochs - 1:
            if args.bias:
                string = "_bias"
            else:
                string = "_no_bias"
            plot_embeddings(video_latent, text_latent, None, directory_name = "val_linear_" + args.loss+string, small_scale=False)

        print(f"Epoch {i+1}/{args.num_epochs}")
        print("train loss", log["train_loss"], "val loss", log["val_loss"])
        print("train", train_accuracies, train_mrr_k, train_mean_sim_score)
        print("val", val_accuracies, val_mrr_k, val_mean_sim_score)
        
        if args.wandb:
            wandb.log(log_dict, step=i)
    








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="X-Clip Video Text Visualization")

    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/xclip-base-patch16-zero-shot", # microsoft/xclip-base-patch16-zero-shot
        help="The name of the model to load.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="The batch size for the data loader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for the random number generator.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of workers for the data loader.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="The learning rate for the optimizer.",
    )
    parser.add_argument(
        "--run_group",
        type=str,
        default="linear_transformation",
        help="The name of the run group for WandB.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="linear_transformation",
        help="The name of the experiment for WandB.",
        required=True,
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=True,
        help="Whether to use WandB for logging.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["cosine", "mse", "nce", "ot"],
        help="The loss function to use for training.",
    )
    parser.add_argument(
        "--bias",
        type=str2bool,
        default=True,
        help="Whether to use bias in the linear transformation.",
    )

  


    args = parser.parse_args()

    main(args)