import os
from sklearn.model_selection import train_test_split
from xclip_utils import OpenXDataset
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np
import argparse
from xclip_utils import load_xclip_model
from pca_utils import plot_embeddings, normalize_embeddings
from measure_utils import check_pairs





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

    video_text_dataset = OpenXDataset(
        video_folder_path = '/scr/yusenluo/RoboCLIP/OpenX/droid', 
        transform=None,
        num_samples=None, 
        random_samples=False, 
        dataset_name='droid',
        debug = False,
        process_frame=False,
        model_name=args.model_name,
    )

    seen_labels = set()
    unique_indices = []
    for idx in tqdm(range(len(video_text_dataset))):
        item = video_text_dataset[idx]
        text_label = item['text']
        if text_label not in seen_labels:
            seen_labels.add(text_label)
            unique_indices.append(idx)
    video_text_dataset.process_frame = True
    unique_dataset = Subset(video_text_dataset, unique_indices)
    train_dataset, val_dataset = train_test_split(unique_dataset, test_size=0.2, random_state=42)

    train_text_feature_total = None
    val_video_feature_total = None

    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for batch in tqdm(training_loader):
        # text, video_frames = batch
        text = batch["text"]
        video_frames = batch["video"]
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


    # for idx in tqdm(range(len(train_dataset))):
    #     # item = train_dataset[idx]
    #     # text_label = item['text']
    #     # video_frames = item['video']
    #     # print(text_label)
    #     import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()




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
        default=4,
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