import os
import h5py
import json
import imageio
import numpy as np
from tqdm import tqdm

from meta_world_name_ann import task_ann
from s3dg import S3D
import torch as th
import torch.nn.functional as F

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

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


h5_embedding_file = h5py.File("metaworld_s3d_embedding.h5", "w")

keys = list(task_ann.keys())
env_names = [task_ann[key] for key in keys]

net = S3D('../s3d_dict.npy', 512)
net.load_state_dict(th.load('../s3d_howto100m.pth'))
net.eval().cuda()

h5_video_file = "/scr/jzhang96/metaworld_gifs_1.h5"
video_h5 = h5py.File(h5_video_file, "r")
# import pdb ; pdb.set_trace()
# data_1 = video_h5['door-open-v2-goal-hidden']['output_gif_10.gif']
# data_1 = np.asarray(data_1)
# data_1 = sample_frames(data_1)
# import imageio
# imageio.mimsave('output.gif', data_1, duration=0.2)
# total_embedding = None
# list_order = []
for i in tqdm(range (len(keys))):
    key = keys[i]
    video_group = video_h5[key]
    videos = []
    for gif_name in video_group.keys():
        video_data = video_group[gif_name]
        video_data = np.asarray(video_data)
        video_data = sample_frames(video_data)
        video_data = np.expand_dims(video_data, axis = 0)
        videos.append(video_data)

    videos = np.concatenate(videos, axis = 0)
    videos = th.tensor(videos).cuda()
    videos = videos.float().permute(0, 4, 1, 2, 3)
    videos = videos / 255.0
    videos_embedding = net(videos)["video_embedding"]

    videos_embedding = videos_embedding.detach().cpu().numpy()
    videos_embedding = normalize_embeddings(videos_embedding, False)

    h5_embedding_file.create_dataset(key, data=videos_embedding)

h5_embedding_file.close()




