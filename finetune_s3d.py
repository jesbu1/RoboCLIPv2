import PIL
from s3dg import S3D
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
from finetune_utils.open_x_loader import TextVideoDataset, EmbeddingsDataset
import wandb
import json
import h5py

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

    s3d = S3D("./s3d_dict.npy", 512)

    if args.finetune:
        s3d.load_state_dict(th.load("./s3d_howto100m.pth"))
    # s3d = s3d.cuda()
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
    train_dataset = TextVideoDataset(args, h5_file, split, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # val_dataset = TextVideoDataset(args, h5_file, split, split="val")
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    for epoch in range(args.num_epochs):
        s3d.train()
        for data in tqdm(train_dataloader):
            a = 0
            video_frames = data["video_frames"]
            # transpose(0,4,1,2,3)
            video_frames = video_frames.permute(0,4,1,2,3).float()
            text = data["text"]
            video_embeddings = s3d(video_frames)["video_embedding"]
            text_embeddings = s3d.text_module(text)["text_embedding"]
            
            # print(video_embeddings.shape)
            # text_embeddings = s3d.text_module(text)
            # import pdb; pdb.set_trace()
            # print(video_frames.shape)
            # print(text)
            # break




if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")


    parser = argparse.ArgumentParser()
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


    args = parser.parse_args()
    print(args)
    main(args)