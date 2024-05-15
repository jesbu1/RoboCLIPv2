import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding, filter_top_embeddings, SthDataset, OpenXDataset
from sklearn.model_selection import train_test_split
import torch as th
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D
import shutil
from torch.utils.data import Subset

import os
from mlp import normalize_embeddings, standradize_embeddings
import argparse


def debug_video_embeddings(sampled_dataset, sampled_video_embeddings, batch_size=1):
    sampled_data_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    debug_embeddings = []
    for i, data in enumerate(sampled_data_loader):
        video_embedding = s3d(th.squeeze(data['video'].float(), dim=1))['video_embedding']
        debug_embeddings.append(video_embedding)
    debug_embeddings = torch.cat(debug_embeddings, dim=0)
    return torch.allclose(debug_embeddings, sampled_video_embeddings)


def find_video(sampled_texts, target_folder):
    for sampled_text in sampled_texts:
        video_id = sampled_text.replace(' ', '_')
        video_path = os.path.join('/scr/yusenluo/RoboCLIP/OpenX/droid', f"{video_id}.gif")

        if os.path.exists(video_path):
            shutil.copy(video_path, target_folder)
            print(f"Copied {video_path} to {target_folder}")
        else:
            print(f"Video file {video_path} does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=bool, default=False)
    args = parser.parse_args()
    filter = False
    if args.filter:
        filter = True
    variance_thresholds = [512, 0.9, 0.95]
    sample_sizes = [10, 20, 50]  # [1, 2, 4, 8, 16, 21]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths = list_webm_files(
        "../20bn-something-something-v2/train"
    )  # '../20bn-something-something-v2'
    # print(len(video_paths))
    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(torch.load("../s3d_howto100m.pth"))
    s3d.eval()

    video_text_dataset = OpenXDataset(
        '/scr/yusenluo/RoboCLIP/OpenX/droid', random_samples=False, dataset_name='droid'
    )
    seen_labels = set()
    unique_indices = []
    for idx in range(len(video_text_dataset)):
        item = video_text_dataset[idx]
        text_label = item['text']
        if text_label not in seen_labels:
            seen_labels.add(text_label)
            unique_indices.append(idx)

    unique_dataset = Subset(video_text_dataset, unique_indices)
    print(len(unique_dataset))
    train_dataset, val_dataset = train_test_split(unique_dataset, test_size=0.2, random_state=42)

    train_data_loader = DataLoader(
        train_dataset, batch_size=10, shuffle=False, num_workers=5
    )
    validate_data_loader = DataLoader(
        val_dataset, batch_size=50, shuffle=False, num_workers=5
    )
    train_video_embeddings, train_text_embeddings, train_embeddings_dataset, mappings = Embedding(
        s3d, train_data_loader
    )
    # for i in range(len(train_embeddings_dataset)):
    #     print(train_dataset[i]['video_id'].replace('_', ' ') == train_dataset[i]['text'])
    #     print(train_embeddings_dataset[i]['video_id'] == train_dataset[i]['video_id'])
    #     print(train_embeddings_dataset[i]['text_label'] == train_dataset[i]['text'])

    validate_video_embeddings, validate_text_embeddings, embeddings_dataset, mappings = Embedding(
        s3d, validate_data_loader
    )
    validate_video_embeddings_normalized = normalize_embeddings(
        validate_video_embeddings
    ).clone()
    validate_text_embeddings_normalized = normalize_embeddings(
        validate_text_embeddings
    ).clone()
    # print("uni", len(unique_indices))
    for size_multiplier in sample_sizes:
        current_sample_size = size_multiplier
        for seed in seeds:
            print(f'Size {size_multiplier} Seed {seed}')
            torch.manual_seed(seed)
            indices = torch.randperm(len(train_video_embeddings))[:size_multiplier]
            sampled_dataset = Subset(train_dataset, indices)
            sampled_data_loader = DataLoader(
                sampled_dataset, batch_size=50, shuffle=False, num_workers=5
            )
            dataset_video_embeddings, dataset_text_embeddings, dataset_embeddings_dataset, mappings = Embedding(
                s3d, sampled_data_loader
            )
            sampled_video = [sampled_dataset[idx]['video'] for idx in range(len(indices.tolist()))]
            sampled_video_1 = [train_dataset[idx]['video'] for idx in indices]
            print(sampled_video_1 == sampled_video)

            sampled_video_embeddings = train_video_embeddings[indices]
            sampled_text_embeddings = train_text_embeddings[indices]
            
            sampled_embeddings_dataset = Subset(train_embeddings_dataset, indices)
            
            print(sampled_video_embeddings.shape)
            print(dataset_video_embeddings.shape)
            for i in range(len(sampled_embeddings_dataset)):
                print(sampled_embeddings_dataset[i]['video_id'] == dataset_embeddings_dataset[i]['video_id'])
                print(sampled_embeddings_dataset[i]['text_label'] == dataset_embeddings_dataset[i]['text_label'])
                print(th.allclose(sampled_embeddings_dataset[i]['video_embedding'], dataset_embeddings_dataset[i]['video_embedding']))
                print(th.allclose(sampled_embeddings_dataset[i]['text_embedding'], dataset_embeddings_dataset[i]['text_embedding']))

            video_embeddings_equal = torch.allclose(dataset_video_embeddings, sampled_video_embeddings)
            text_embeddings_equal = torch.allclose(dataset_text_embeddings, sampled_text_embeddings)

            print(f"Seed {seed}, Size {size_multiplier}:")
            sampled_texts = [train_dataset[idx]['text'] for idx in indices]
            sampled_texts_1 = [train_embeddings_dataset[idx]['text_label'] for idx in indices]
            sampled_texts_2 = [sampled_embeddings_dataset[idx]['text_label'] for idx in range(len(indices.tolist()))]
            print(sampled_texts == sampled_texts_1)
            print(sampled_texts_1 == sampled_texts_2)
            print(sampled_texts == sampled_texts_2)
            print("Sampled Texts:", sampled_texts)
            print(f"Video Embeddings Equal: {video_embeddings_equal}")
            print(f"Text Embeddings Equal: {text_embeddings_equal}")
            # Test_Model(validate_video_embeddings_normalized.to(device), validate_text_embeddings_normalized.to(device),
            #            mappings, 0, None, 'Validate')
            

            # train_video_embeddings_normalized = normalize_embeddings(
            #     sampled_video_embeddings
            # ).clone()
            # train_text_embeddings_normalized = normalize_embeddings(
            #     sampled_text_embeddings
            # ).clone()
