import PIL
# from S3D_HowTo100M.s3dg import S3D
import json
import numpy as np
import os
import cv2
# from pca import plot_embeddings_3d, plot_embeddings, check_pairs
# from mlp import mlp_eval, normalize_embeddings, standradize_embeddings
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import visualization
from visualization.Dataload import VideoTextDataset, Embedding, list_webm_files, OpenXDataset
import argparse
from finetune_utils.s3d_loss import MILNCELoss
from finetune_utils.open_x_loader import TextVideoDataset, EmbeddingsDataset, XCLIPDataset
import wandb
import json
import h5py
import re
from finetune_utils.s3d_loss import MILNCELoss
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import numpy as np
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")









def main(args):



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    WANDB_ENTITY_NAME = "henry_32"
    WANDB_PROJECT_NAME = "roboclip-v2"




    if args.experiment_name is None:
        exp_name = "xclip_droid_finetune"
    else:
        exp_name = args.experiment_name
#     wandb.init(
#         # set the wandb project where this run will be logged
#         entity=WANDB_ENTITY_NAME,
#         project=WANDB_PROJECT_NAME,
#         resume= exp_name,
#         group=args.run_group,
#         # track hyperparameters and run metadata
#         config=args,
# )


    xclip_tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # xclip_tokenizer.to("cuda")
    xclip_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = torch.compile(xclip_model).cuda()


    

    if args.debug:
        # dataset_name = "droid_100_torch_train.h5"
        dataset_name = "droid_100_torch_compress_test_train.h5"
        
    else:
        dataset_name = args.dataset_name + "_torch_compress_train" + ".h5"

    h5_file = h5py.File(os.path.join(args.dataset_path, dataset_name), 'r')
    split_file = args.dataset_name + "_torch_split.json"
    split = json.load(open(os.path.join("finetune_utils", split_file), "r"))
    print("Dataset", args.dataset_name, "train", len(split["train"]), "val", len(split["val"]))

    
    # load the dataset
    train_dataset = XCLIPDataset(
                                args, 
                                h5_file, 
                                split, 
                                split="train", 
                                tokenizer = xclip_tokenizer,
                                )

    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            )

    val_dataset = XCLIPDataset(
                            args, 
                            h5_file, 
                            split, 
                            split="val",
                            tokenizer = xclip_tokenizer,
                            )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
        )

    
    criterion = MILNCELoss()

    optimizer = torch.optim.Adam(xclip_model.parameters(), lr=args.lr)

    grad_scaler = GradScaler()

    step = 0

    for epoch in range(args.num_epochs):
        xclip_model.train()
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            video_frames = data["video_frames"]
            video_frames = video_frames.permute(0,1,4,2,3).float().cuda()
            text = data["text"]
            
            if args.use_amp:
                with autocast():
                    import ipdb ; ipdb.set_trace()
                    # text_emb = xclip_tokenizer(text, padding=True, return_tensors="pt")
                    # video_emb = xclip_processor(videos=video_frames, return_tensors="pt")
                    # video_embeddings = xclip_model.get_video_features(**video_emb)
                    # text_embeddings = xclip_model.get_text_features(**text_emb)
                    video_features, _ = xclip_model.encode_video(video_frames) 


                    loss = criterion(video_embeddings, text_embeddings)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            # else:
            #     video_embeddings = s3d(video_frames)["video_embedding"]
            #     text_embeddings = s3d.text_module(text_emb)["text_embedding"]
            #     loss = criterion(video_embeddings, text_embeddings)
            #     loss.backward()
            #     optimizer.step()
            wandb.log({"train_loss": loss.item()}, step=step)
            step += 1

        with torch.no_grad():
            xclip_model.eval()
            eval_loss = 0
            for data in tqdm(val_dataloader):
                video_frames = data["video_frames"]
                video_frames = video_frames.permute(0,4,1,2,3).float().cuda()
                text = data["text"]
                text_emb = xclip_tokenizer(text, padding=True, return_tensors="pt").cuda()
                if args.use_amp:
                    with autocast():
                        video_embeddings = xclip_model.get_video_features(video_frames)
                        text_embeddings = xclip_model.get_text_features(**text_emb)
                        loss = criterion(video_embeddings, text_embeddings)
                    eval_loss += loss.item()
                # else:
                #     video_embeddings = s3d(video_frames)["video_embedding"]
                #     text_embeddings = s3d.text_module(text_emb)["text_embedding"]
                #     loss = criterion(video_embeddings, text_embeddings)
                #     eval_loss += loss.item()
            eval_loss /= len(val_dataloader)
            wandb.log({"eval_loss": eval_loss}, step=step)

        if epoch % args.save_freq == args.save_freq - 1:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            folder = os.path.join(args.save_path, args.experiment_name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            name = os.path.join(folder, f"xclip_finetune_{epoch}.pth")
            torch.save(xclip_model.state_dict(), name)

        

        









if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="s3d_finetune", help="Name of the experiment")
    parser.add_argument("--run_group", type=str, default="s3d_finetune", help="Name of the run group")
    parser.add_argument("--finetune", type=str2bool, default=False, help="Whether to finetune the model, if False will train from scratch")
    parser.add_argument("--dataset_name", type=str, default="droid", choices=["droid", "droid_100", "bridge", "fractal"], help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--dataset_path", type=str, default="/scr/jzhang96/", help="Path to the dataset")
    parser.add_argument("--preprocess", type=str2bool, default=True, help="Whether to preprocess the dataset")
    parser.add_argument("--ds_frames", type=int, default=32, help="Number of frames to sample from the video")
    parser.add_argument("--debug", type=str2bool, default=True, help="Debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="/scr/jzhang96/s3d_finetune", help="Path to save the model")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency to save the model")
    parser.add_argument("--use_amp", type=str2bool, default=True, help="Whether to use automatic mixed precision")


    args = parser.parse_args()
    print(args)
    main(args)