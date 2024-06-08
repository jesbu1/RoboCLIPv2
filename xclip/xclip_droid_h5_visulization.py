import os
from sklearn.model_selection import train_test_split
from xclip_utils import OpenXDataset, DroidH5Dataset
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np
import argparse
from xclip_utils import load_xclip_model
from pca_utils import plot_embeddings, normalize_embeddings
from measure_utils import check_pairs
os.environ["TOKENIZERS_PARALLELISM"] = "false"





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

    video_text_dataset = DroidH5Dataset(
        h5_path = "/scr/jzhang96/droid_sth_dataset.hdf5",
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

    train_text_feature_total = None
    train_video_feature_total = None
    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    for batch in tqdm(training_loader):
        # text, video_frames = batch
        text = batch["text"]
        video_frames = batch["video"]
        video_frames = video_frames.to(device)

        text_tokens = xclip_tokenizer(text, padding=True, return_tensors="pt").to(device)
        text_features = xclip_model.get_text_features(**text_tokens)
        video_features = xclip_model.get_video_features(video_frames)

        if train_text_feature_total is None:
            train_text_feature_total = text_features.cpu().detach().numpy()
            train_video_feature_total = video_features.cpu().detach().numpy()
        else:
            train_text_feature_total = np.concatenate((train_text_feature_total, text_features.cpu().detach().numpy()), axis=0)
            train_video_feature_total = np.concatenate((train_video_feature_total, video_features.cpu().detach().numpy()), axis=0)
        

    train_video_feature_total = normalize_embeddings(train_video_feature_total)
    train_text_feature_total = normalize_embeddings(train_text_feature_total)
    print("length of train_video_feature_total: ", len(train_video_feature_total))

    sim_score = train_video_feature_total @ train_text_feature_total.T
    mean_sim_score = np.mean(sim_score)
    accuracies, mrr_k = check_pairs(train_video_feature_total, train_text_feature_total, None, small_scale=False)
    for k, v in accuracies.items():
        print(f"Training Top {k} accuracy: {v}")
    for k, v in mrr_k.items():
        print(f"Training Top {k} MRR: {v}")
    print(f"Train Mean similarity score: {mean_sim_score}")

    simi_score = torch.mean(torch.diag(torch.matmul(torch.tensor(train_video_feature_total), torch.tensor(train_text_feature_total).t())))
    print(f"Torch Train Mean similarity score: {simi_score}")



    val_text_feature_total = None
    val_video_feature_total = None

    for batch in tqdm(val_loader):
        # text, video_frames = batch
        text = batch["text"]
        video_frames = batch["video"]
        video_frames = video_frames.to(device)

        text_tokens = xclip_tokenizer(text, padding=True, return_tensors="pt").to(device)
        text_features = xclip_model.get_text_features(**text_tokens)
        video_features = xclip_model.get_video_features(video_frames)

        if val_text_feature_total is None:
            val_text_feature_total = text_features.cpu().detach().numpy()
            val_video_feature_total = video_features.cpu().detach().numpy()
        else:
            val_text_feature_total = np.concatenate((val_text_feature_total, text_features.cpu().detach().numpy()), axis=0)
            val_video_feature_total = np.concatenate((val_video_feature_total, video_features.cpu().detach().numpy()), axis=0)

    val_video_feature_total = normalize_embeddings(val_video_feature_total)
    val_text_feature_total = normalize_embeddings(val_text_feature_total)
    val_sim_score = val_video_feature_total @ val_text_feature_total.T
    val_sim_score = np.mean(val_sim_score)
    accuracies, mrr_k = check_pairs(val_video_feature_total, val_text_feature_total, None, small_scale=False)
    for k, v in accuracies.items():
        print(f"Validation Top {k} accuracy: {v}")
    for k, v in mrr_k.items():
        print(f"Validation Top {k} MRR: {v}")
    print(f"Val Mean similarity score: {val_sim_score}")

    simi_score = torch.mean(torch.diag(torch.matmul(torch.tensor(val_video_feature_total), torch.tensor(val_text_feature_total).t())))
    print(f"Torch Train Mean similarity score: {simi_score}")




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