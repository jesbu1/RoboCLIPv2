import os
import h5py
import json
import imageio
import numpy as np
from tqdm import tqdm
id_tasks = json.load(open("id_task.json", "r"))
keys = list(id_tasks.keys())
env_names = [id_tasks[key] for key in keys]
folder_prefix = "/scr/jzhang96/metaworld_generate_gifs/"

hr_file = h5py.File("/scr/jzhang96/metaworld_gifs_1.h5", "w")

for i in tqdm(range (len(keys))):
    task_name = keys[i]
    task_idx = env_names[i]
    hr_file.create_group(task_name)
    task_folder = os.path.join(folder_prefix, task_idx)
    gif_names = os.listdir(task_folder)
    num = 0
    for gif_name in gif_names:
        file_name = os.path.join(task_folder, gif_name)
        reader = imageio.get_reader(file_name)
        frames = []
        for frame in reader:
            frames.append(frame[:,:,:3])
        frames = np.array(frames)
        frames = frames[:,240 - 125:240 + 125, 320 - 125:320 + 125, :]
        hr_file[task_name].create_dataset(gif_name, data=frames)
        num += 1
    print(task_name, num)
hr_file.close()
        

        





# for task_idx in tqdm(task_id.keys()):
#     task_name = task_id[task_idx]
#     task_annotation = task_ann[task_name]
#     task_folder = os.path.join(folder_prefix, task_idx)

#     if check_folder(task_folder, num_videos=1):

#         group = h5_file.create_group(task_idx)
#         group.attrs["task_name"] = task_name
#         group.attrs["task_annotation"] = task_annotation
#         text_tokens = tokenizer([task_annotation], return_tensors="pt").to("cuda")
#         text_features = model.get_text_features(**text_tokens)
#         group.create_dataset("xclip_text_feature", data=text_features.squeeze(0).cpu().detach().numpy())

#         video_files = os.listdir(task_folder)

#         for i, video_file in enumerate(video_files):
#             video_path = os.path.join(task_folder, video_file)
#             # frames = video2array(video_path)
#             # read gif imageio
            
#             reader = imageio.get_reader(video_path)
#             frames = []
#             for frame in reader:
#                 frames.append(frame[:,:,:3])
#             latents_frames = preprocess_human_demo_xclip(frames)
#             latents_frames = adjust_frames_xclip(latents_frames)
#             latents_frames = [latents_frames[i] for i in range (latents_frames.shape[0])]
#             latents_frames = processor(videos=latents_frames, return_tensors="pt")
#             latents_frames = latents_frames["pixel_values"].cuda()
#             latents_frames = model.get_video_features(latents_frames).squeeze(0).cpu().detach().numpy()

#             frames = np.array(frames)
#             data_group = group.create_group(str(i))
#             # data_group.create_dataset("frames", data=frames)
#             data_group.create_dataset("xclip_video_feature", data=latents_frames)

# h5_file.close()
