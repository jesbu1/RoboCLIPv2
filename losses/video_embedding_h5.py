import os
import h5py
import json
import imageio
import numpy as np
from tqdm import tqdm

from meta_world_name_ann import task_ann
from s3dg import S3D
import torch as th

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



# h5_embedding_file = h5py.File("metaworld_s3d_embedding.h5", "w")

# keys = list(task_ann.keys())
# env_names = [task_ann[key] for key in keys]

# net = S3D('../s3d_dict.npy', 512)
# net.load_state_dict(th.load('../s3d_howto100m.pth'))
# net.eval().cuda()

h5_video_file = "/scr/jzhang96/metaworld_gifs_1.h5"
video_h5 = h5py.File(h5_video_file, "r")
import pdb ; pdb.set_trace()
# total_embedding = None
# list_order = []
# for i in tqdm(range (len(keys))):
#     key = keys[i]
#     video_group = video_h5[key]
#     videos = None
#     for gif_name in video_group.keys():
#         video_data = video_group[gif_name]
#         video_data = np.asarray(video_data)
#         video_data = sample_frames(video_data)
#         video_data = np.expand_dims(video_data, axis = 0)
#         if videos is None:
#             videos = video_data
#         else:
#             videos = np.concatenate([videos, video_data], axis = 0)
#     videos = th.tensor(videos).cuda()
#     videos = videos.float().permute(0, 4, 1, 2, 3)
#     videos = videos / 255.0
#     videos_embedding = net(videos)["video_embedding"]
#     videos_embedding = videos_embedding.detach().cpu().numpy()
#     # if i == 0:
#     #     total_embedding = videos_embedding
#     # else:
#     #     total_embedding = np.concatenate([total_embedding, videos_embedding], axis = 0)
#     # list_order.append(key)
#     h5_embedding_file.create_dataset(key, data=videos_embedding)

# h5_embedding_file.close()




