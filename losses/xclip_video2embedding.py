# this file is to save the video embeddings to a h5 file. 
# The video embeddings are extracted from the video gifs using the xclip model. 
# The embeddings are not normalized in this file.

# the embeddings include 4 parts:
# 1. Expert Videos
# 2. Time Shuffle Videos
# 3. Time shortened Videos
# 4. Augmentation Videos

import os
import h5py
import json
import imageio
import numpy as np
from tqdm import tqdm
from meta_world_name_ann import task_ann
import torch as th
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import copy
from triplet_utils import AugmentationPipeline


def sample_frames(array, num_frames = 32):
    if len(array) > num_frames:
        indices = np.linspace(0, len(array) - 1, num_frames, dtype=int)
        return array[indices]
    else:
        more_frames = num_frames - len(array)
        last_frame = array[-1:]
        for i in range(more_frames):
            array = np.concatenate([array, last_frame], axis=0)
    return array

def sample_frames_front(array, num_frames = 32):
    if len(array) > num_frames:
        indices = np.linspace(0, len(array) - 1, num_frames, dtype=int)
        return array[indices]
    else:
        more_frames = num_frames - len(array)
        first_frame = array[:1]
        for i in range(more_frames):
            array = np.concatenate([first_frame, array], axis=0)
    return array


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()

def shuffle_time_func(array):
        random_index = np.random.permutation(len(array))
        return array[random_index]



h5_embedding_file = h5py.File("/scr/jzhang96/metaworld_xclip_all_embedding_15.h5", "r+")

keys = list(task_ann.keys())
env_names = [task_ann[key] for key in keys]

model_name = "microsoft/xclip-base-patch16-zero-shot"
xclip_tokenizer = AutoTokenizer.from_pretrained(model_name)
xclip_net = AutoModel.from_pretrained(model_name).cuda()
xclip_processor = AutoProcessor.from_pretrained(model_name)
xclip_net.eval()

h5_video_file = "/scr/jzhang96/metaworld_gifs_1.h5"
video_h5 = h5py.File(h5_video_file, "r")
augumentation_pipeline = AugmentationPipeline(device="cuda", strength='normal')
print("keys", keys)

# print("phase 1 embedding GT videos")

# # create group name GT_videos

# group_name = "GT_Videos"
# h5_embedding_file.create_group(group_name)
# for i in range (len(keys)):
#     print(i)
#     key = keys[i]
#     video_group = video_h5[key]
#     videos = []
#     for gif_name in tqdm(list(video_group.keys())[:15]):
#         video_data = video_group[gif_name]
#         video_data = np.asarray(video_data)
#         video_data = sample_frames(video_data)
#         video_data = video_data[:, 13:237, 13:237, :]
#         video_data = xclip_processor(videos = list(video_data), return_tensors="np").pixel_values
#         videos.append(video_data)
#     videos = np.concatenate(videos, axis=0)
    
#     videos = th.tensor(videos).cuda()
#     with th.no_grad():
#         video_embeddings = xclip_net.get_video_features(videos)
#     video_embeddings = video_embeddings.cpu().numpy()
#     h5_embedding_file[group_name].create_dataset(key, data=video_embeddings)

# print("phase 2 embedding Time Shuffle Videos") # each video create 50 time shuffle videos
# group_name = "Time_Shuffle_Videos"
# h5_embedding_file.create_group(group_name)
# for i in range (len(keys)):
#     key = keys[i]
#     video_group = video_h5[key]
#     certain_video_embeddings = []
#     for gif_name in tqdm(list(video_group.keys())[:15]):
#         video_data = video_group[gif_name]
#         video_data = np.asarray(video_data)
#         video_data = video_data[:, 13:237, 13:237, :]
#         videos = []
#         for _ in range(50):
#             curr_video_data = shuffle_time_func(copy.deepcopy(video_data))
#             curr_video_data = sample_frames(curr_video_data)
#             curr_video_data = xclip_processor(videos = list(curr_video_data), return_tensors="np").pixel_values

#             videos.append(curr_video_data)
#         videos = np.concatenate(videos, axis=0)
#         videos = th.tensor(videos).cuda()
#         with th.no_grad():
#             video_embeddings = xclip_net.get_video_features(videos)
#         video_embeddings = video_embeddings.cpu().numpy()
#         certain_video_embeddings.append(video_embeddings)
#     certain_video_embeddings = np.concatenate(certain_video_embeddings, axis=0)
#     h5_embedding_file[group_name].create_dataset(key, data=certain_video_embeddings)

print("phase 3 embedding Time Shortened Videos") # each video create embedding and progress
group_name = "Time_Shortened_Videos"
if group_name not in list(h5_embedding_file.keys()):
    h5_embedding_file.create_group(group_name)

for i in range (len(keys)):
    print(i)
    key = keys[i]
    print(key)
    if key not in list(h5_embedding_file[group_name].keys()):
        
        video_group = video_h5[key]
        certain_video_embeddings = []
        certain_progress = []
        for gif_name in tqdm(list(video_group.keys())[:15]):
            video_data = video_group[gif_name]
            video_data = np.asarray(video_data)
            video_data = video_data[:, 13:237, 13:237, :]
            videos = []
            progress =[]

            video_length = len(video_data)
            no_use_length = [1, video_length - 2, video_length - 3]
            for i in range (1, video_length):
                if i in no_use_length:
                    continue
                curr_video_data = video_data[:i]
                curr_video_data = sample_frames_front(curr_video_data)
                curr_video_data = xclip_processor(videos = list(curr_video_data), return_tensors="np").pixel_values
                videos.append(curr_video_data)
                progress.append(i/(video_length-1))
            videos = np.concatenate(videos, axis=0)
            videos = th.tensor(videos).cuda()
            print("videos shape", videos.shape)
            with th.no_grad():
                if videos.shape[0] > 40 and videos.shape[0] <= 80:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2], axis=0)
                elif videos.shape[0] >= 80 and videos.shape[0] <= 120:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2, embedding_3], axis=0)
                elif videos.shape[0] > 120 and videos.shape[0] <= 160:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:120]).cpu().detach().numpy()
                    embedding_4 = xclip_net.get_video_features(videos[120:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2, embedding_3, embedding_4], axis=0)
                elif videos.shape[0] > 160 and videos.shape[0] <= 200:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:120]).cpu().detach().numpy()
                    embedding_4 = xclip_net.get_video_features(videos[120:160]).cpu().detach().numpy()
                    embedding_5 = xclip_net.get_video_features(videos[160:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2, embedding_3, embedding_4, embedding_5], axis=0)
                elif videos.shape[0] > 200 and videos.shape[0] <= 240:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:120]).cpu().detach().numpy()
                    embedding_4 = xclip_net.get_video_features(videos[120:160]).cpu().detach().numpy()
                    embedding_5 = xclip_net.get_video_features(videos[160:200]).cpu().detach().numpy()
                    embedding_6 = xclip_net.get_video_features(videos[200:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, embedding_6], axis=0)
                elif videos.shape[0] > 240 and videos.shape[0] <= 280:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:120]).cpu().detach().numpy()
                    embedding_4 = xclip_net.get_video_features(videos[120:160]).cpu().detach().numpy()
                    embedding_5 = xclip_net.get_video_features(videos[160:200]).cpu().detach().numpy()
                    embedding_6 = xclip_net.get_video_features(videos[200:240]).cpu().detach().numpy()
                    embedding_7 = xclip_net.get_video_features(videos[240:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, embedding_6, embedding_7], axis=0)
                elif videos.shape[0] > 280 and videos.shape[0] <= 320:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:120]).cpu().detach().numpy()
                    embedding_4 = xclip_net.get_video_features(videos[120:160]).cpu().detach().numpy()
                    embedding_5 = xclip_net.get_video_features(videos[160:200]).cpu().detach().numpy()
                    embedding_6 = xclip_net.get_video_features(videos[200:240]).cpu().detach().numpy()
                    embedding_7 = xclip_net.get_video_features(videos[240:280]).cpu().detach().numpy()
                    embedding_8 = xclip_net.get_video_features(videos[280:]).cpu().detach().numpy()
                    video_embeddings = np.concatenate([embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, embedding_6, embedding_7, embedding_8], axis=0)
                elif videos.shape[0] > 320 and videos.shape[0] <= 360:
                    embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().detach().numpy()
                    embedding_2 = xclip_net.get_video_features(videos[40:80]).cpu().detach().numpy()
                    embedding_3 = xclip_net.get_video_features(videos[80:120]).cpu().detach().numpy()
                    embedding_4 = xclip_net.get_video_features(videos[120:160]).cpu().detach().numpy()
                    embedding_5 = xclip_net.get_video_features(videos[160:200]).cpu().detach().numpy()
                    embedding_6 = xclip_net.get_video_features(videos[200:240]).cpu().detach().numpy()
                    embedding_7 = xclip_net.get_video_features(videos[240:280]).cpu().detach().numpy()
                    embedding_8 = xclip_net.get_video_features(videos[280:320]).cpu().detach().numpy()
                    embedding_9 = xclip_net.get_video_features(videos[320:]).cpu().detach().numpy()
                else:
                    video_embeddings = xclip_net.get_video_features(videos)
                    video_embeddings = video_embeddings.cpu().numpy()
            certain_video_embeddings.append(video_embeddings)
            certain_progress.append(progress)
        certain_video_embeddings = np.concatenate(certain_video_embeddings, axis=0)
        certain_progress = np.concatenate(certain_progress, axis=0)
        h5_embedding_file[group_name].create_dataset(key, data=certain_video_embeddings)
        h5_embedding_file[group_name].create_dataset(key+"_progress", data=certain_progress)

print("phase 4 embedding Augmentation Videos") # each video create 80 augmentation videos
group_name = "Augmentation_Videos"
h5_embedding_file.create_group(group_name)

for i in range (len(keys)):
    print(i)
    key = keys[i]
    video_group = video_h5[key]
    certain_video_embeddings = []
    for gif_name in tqdm(list(video_group.keys())[:15]):
        video_data = video_group[gif_name]
        video_data = np.asarray(video_data)
        video_data = video_data[:, 13:237, 13:237, :]

        videos = []
        for _ in range(80):
            curr_video_data = copy.deepcopy(video_data)
            curr_video_data = sample_frames(curr_video_data)
            curr_video_data = curr_video_data.transpose(0, 3, 1, 2)
            curr_video_data = np.expand_dims(curr_video_data, axis=0)
            curr_video_data = th.tensor(curr_video_data).cuda().float()/255
            curr_video_data = augumentation_pipeline(curr_video_data)
            # save the augmented video as gif
            curr_video_data = curr_video_data.cpu().numpy()
            curr_video_data = curr_video_data.squeeze(0).transpose(0, 2, 3, 1)
            curr_video_data = curr_video_data *255
            curr_video_data = curr_video_data.astype(np.uint8)
            
            curr_video_data = xclip_processor(videos = list(curr_video_data), return_tensors="np").pixel_values
            videos.append(curr_video_data)
        videos = np.concatenate(videos, axis=0)
        videos = th.tensor(videos).cuda()
        with th.no_grad():
            if videos.shape[0] > 40:
                embedding_1 = xclip_net.get_video_features(videos[:40]).cpu().numpy()
                embedding_2 = xclip_net.get_video_features(videos[40:]).cpu().numpy()
                video_embeddings = np.concatenate([embedding_1, embedding_2], axis=0)
            else:
                video_embeddings = xclip_net.get_video_features(videos)
                video_embeddings = video_embeddings.cpu().numpy()
        certain_video_embeddings.append(video_embeddings)
    certain_video_embeddings = np.concatenate(certain_video_embeddings, axis=0)
    h5_embedding_file[group_name].create_dataset(key, data=certain_video_embeddings)

h5_embedding_file.close()










h5_embedding_file.close()



        
        
        # video_data = xclip_processor(videos = list(video_data), return_tensors="np").pixel_values
        # videos.append(video_data)







