import os
import torch
import argparse
from xclip_utils import list_webm_files, load_xclip_model, TextVideoDataset
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XCLIPTextModel, XCLIPVisionModel
from pca_utils import plot_embeddings, normalize_embeddings
from measure_utils import check_pairs



# inputs = processor(text=["playing sports", "eating spaghetti", "go shopping"], videos=list(video), return_tensors="pt", padding=True)

# # forward pass
# with torch.no_grad():
#     outputs = model(**inputs)

# probs = outputs.logits_per_video.softmax(dim=1)
# probs

def main(args):

    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the X-Clip model
    xclip_tokenizer, xclip_processor, xclip_model = load_xclip_model(args.model_name, device)

    # Load the video paths
    training_video_paths = os.path.join(args.video_folder_path, "train")
    validation_video_paths = os.path.join(args.video_folder_path, "validation")

    # get text data
    text_label_dict = json.load(open(os.path.join(args.text_path, "video_text_dict.json")))

    # Load the dataset
    training_dataset = TextVideoDataset(
        training_video_paths, 
        text_label_dict,
        target_frame_count=args.frame_num,
        model_name=args.model_name,
        device=device,
    )
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = TextVideoDataset(
        validation_video_paths, 
        text_label_dict,
        target_frame_count=args.frame_num,
        model_name=args.model_name,
        device=device,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    text_feature_total = None
    video_feature_total = None

    for batch in tqdm(training_loader):
        text, video_frames = batch
        video_frames = video_frames.to(device)

        text_tokens = xclip_tokenizer(text, padding=True, return_tensors="pt").to(device)
        text_features = xclip_model.get_text_features(**text_tokens)
        video_features = xclip_model.get_video_features(video_frames)

        if text_feature_total is None:
            text_feature_total = text_features.cpu().detach().numpy()
            video_feature_total = video_features.cpu().detach().numpy()
        else:
            text_feature_total = np.concatenate((text_feature_total, text_features.cpu().detach().numpy()), axis=0)
            video_feature_total = np.concatenate((video_feature_total, video_features.cpu().detach().numpy()), axis=0)

    video_feature_total = normalize_embeddings(video_feature_total)
    text_feature_total = normalize_embeddings(text_feature_total)


    accuracies, mrr_k = check_pairs(video_feature_total, text_feature_total, None, small_scale=False)
    for k, v in accuracies.items():
        print(f"Training Top {k} accuracy: {v}")
    for k, v in mrr_k.items():
        print(f"Training Top {k} MRR: {v}")
        
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # embeddings = np.concatenate((video_feature_total, text_feature_total), axis=0)
    # embeddings = normalize_embeddings(embeddings)
    # video_feature_total = embeddings[:len(video_feature_total)]
    # text_feature_total = embeddings[len(video_feature_total):]

    # # total_feature = np.concatenate((text_feature_total, video_feature_total), axis=0)
    print("length of video_feature_total: ", video_feature_total.shape)
    print("length of text_feature_total: ", text_feature_total.shape)
    plot_embeddings(video_feature_total, text_feature_total, "xclip_imgs", "xclip_pca_2d_norm_seperate_together.png")
    





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="X-Clip Video Text Visualization")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/xclip-base-patch16-zero-shot", # microsoft/xclip-base-patch16-zero-shot
        help="The name of the model to load.",
    )
    parser.add_argument(
        "--video_folder_path",
        type=str,
        default="/scr/yusenluo/RoboCLIP/20bn-something-something-v2",
        help="The path to the folder contains the dataset(webm).",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default="./",
        help="The path to the text file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
        default=8,
        help="The number of workers for the data loader.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=32,
        help="The number of frames to sample from the video.",
    )
    args = parser.parse_args()

    main(args)