'''
generate a json file with dictionary, the key is video id, the value is the text
''' 
import os
import json
import h5py
from measure_utils import check_pairs
import numpy as np
import torch
from xclip_utils import DroidH5LatentDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

new_h5_file = h5py.File("/scr/jzhang96/droid_sth_dataset_latent.hdf5", "r")


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)





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
train_dataset, val_dataset = train_test_split(unique_dataset, test_size=0.2, random_state=seed)

total_video_latent = None
total_text_latent = None
for data in tqdm(val_dataset):
    # text = np.expand_dims(data["text_latent"], axis=0)
    # video = np.expand_dims(data["video_latent"], axis=0)
    video = data["video_latent"]
    text = data["text_latent"]
    total_video_latent = np.expand_dims(video, axis=0) if total_video_latent is None else np.concatenate((total_video_latent, np.expand_dims(video, axis=0)), axis=0) 
    total_text_latent = np.expand_dims(text, axis=0) if total_text_latent is None else np.concatenate((total_text_latent, np.expand_dims(text, axis=0)), axis=0)
    # if total_video_latent is None:
    #     total_video_latent = video
    #     total_text_latent = text
    # else:
    #     total_video_latent = np.concatenate((total_video_latent, video), axis=0)
    #     total_text_latent = np.concatenate((total_text_latent, text), axis=0)

# total_video_latent = total_video_latent / np.linalg.norm(total_video_latent, axis=1, keepdims=True)
# total_text_latent = total_text_latent / np.linalg.norm(total_text_latent, axis=1, keepdims=True)

accuracies, mrr_k = check_pairs(total_video_latent, total_text_latent, None, small_scale=False)
print(accuracies)
print(mrr_k)
mean_sim_score = np.mean(total_video_latent @ total_text_latent.T)
print(mean_sim_score)
# check_pairs

#     reduced_video_embeddings: np.array,
#     reduced_text_embeddings: np.array,
#     mappings,
#     small_scale=True,