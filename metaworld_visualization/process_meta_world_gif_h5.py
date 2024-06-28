'''
convert generated video to h5 file
'''

import os
import sys
# add the "../" directory to the sys.path
parent_dir_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
parent_dir_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xclip/'))
parent_dir_3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../metaworld_generation/'))
sys.path.append(parent_dir_1)
sys.path.append(parent_dir_2)
sys.path.append(parent_dir_3)
import h5py
import json
from meta_world_name_ann import task_ann
from xclip_vis_metaworld_video import get_models, video2array, check_folder
from tqdm import tqdm
import numpy as np
from xclip_utils import preprocess_human_demo_xclip, adjust_frames_xclip
import imageio

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "microsoft/xclip-base-patch16-zero-shot"
model, tokenizer, processor = get_models(model_name)
model.cuda()
file_name = "../metaworld_generation/task_id.json"
with open(file_name, 'r') as f:
    task_id = json.load(f)

h5_file = h5py.File("/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5", "w")

folder_prefix = "/scr/jzhang96/metaworld_generate_gifs/"
for task_idx in tqdm(task_id.keys()):
    task_name = task_id[task_idx]
    task_annotation = task_ann[task_name]
    task_folder = os.path.join(folder_prefix, task_idx)

    if check_folder(task_folder, num_videos=1):

        group = h5_file.create_group(task_idx)
        group.attrs["task_name"] = task_name
        group.attrs["task_annotation"] = task_annotation
        text_tokens = tokenizer([task_annotation], return_tensors="pt").to("cuda")
        text_features = model.get_text_features(**text_tokens)
        group.create_dataset("xclip_text_feature", data=text_features.squeeze(0).cpu().detach().numpy())

        video_files = os.listdir(task_folder)

        for i, video_file in enumerate(video_files):
            video_path = os.path.join(task_folder, video_file)
            # frames = video2array(video_path)
            # read gif imageio
            
            reader = imageio.get_reader(video_path)
            frames = []
            for frame in reader:
                frames.append(frame[:,:,:3])
            latents_frames = preprocess_human_demo_xclip(frames)
            latents_frames = adjust_frames_xclip(latents_frames)
            latents_frames = [latents_frames[i] for i in range (latents_frames.shape[0])]
            latents_frames = processor(videos=latents_frames, return_tensors="pt")
            latents_frames = latents_frames["pixel_values"].cuda()
            latents_frames = model.get_video_features(latents_frames).squeeze(0).cpu().detach().numpy()

            frames = np.array(frames)
            data_group = group.create_group(str(i))
            # data_group.create_dataset("frames", data=frames)
            data_group.create_dataset("xclip_video_feature", data=latents_frames)

h5_file.close()
