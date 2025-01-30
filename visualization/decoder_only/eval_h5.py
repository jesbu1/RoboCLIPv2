from clip_utils import load_model
import torch
import cv2 
import numpy as np
from PIL import Image
import h5py
import os
from tqdm import tqdm
import json
# from new_task_annotation_v2 import train_gt_annotation, eval_gt_annotation

# def main(model_name, video_base_path):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, processor, tokenizer = load_model(model_name)
#     model = model.to(device)

#     # 输入/输出文件路径
#     video_path = "/scr/jzhang96/metaworld_traj_15_demos.h5"
#     video_file = h5py.File(video_path, "r")
#     metaworld_dataset_path = "metaworld_GT_eval_v2.h5"  # 修改文件名
#     data_file = h5py.File(metaworld_dataset_path, "w")

#     total_num = 0

#     for key_name in tqdm(video_file.keys()):
#         video_dataset = video_file[key_name]
#         env_id = video_dataset["env_id"][()][-1].decode("utf-8")  # 获取任务名称

#         if env_id in data_file.keys():
#             continue

#         # 读取图像数据
#         imgs = video_dataset["img"][:]
#         frame_list = []
        
#         for i in range(imgs.shape[0]):
#             img = imgs[i]
#             # 中心裁剪 224x224
#             img_cropped = img[240-112:240+112, 320-112:320+112, :]
#             frame_list.append(img_cropped)

#         # 存储裁剪后的图像
#         img_array = np.array(frame_list, dtype=np.uint8)  # 确保数据格式为 uint8
#         data_file.create_dataset(env_id, data=img_array)

#         total_num += 1

#     print(f"Total processed: {total_num}")

#     # 存储任务文本
#     key_list = []
#     ann_list = []

#     for key in train_gt_annotation:
#         key_list.append(key)
#         ann_list.append(train_gt_annotation[key])

#     for key in eval_gt_annotation:
#         key_list.append(key)
#         ann_list.append(eval_gt_annotation[key])

#     key_list += ["clean-table"]
#     ann_list += ["Cleaning the table"]

#     # 存储文本数据
#     text_group = data_file.create_group("text_annotations")
    

#     for i, key in enumerate(key_list):
#         key_name = key + "_text"
        
#         # 解决 "name already exists" 问题
#         if key_name in text_group:
#             del text_group[key_name]  # 先删除已存在的数据集
        
#         text_group.create_dataset(key_name, data=np.string_(ann_list[i]))  # 存储字符串


#     data_file.close()

# if __name__ == "__main__":
#     video_base_path = "/scr/jzhang96/metaworld_25_for_clip/"
#     main("liv", video_base_path)

# HDF5 文件路径
# h5_file_path = "metaworld_GT_eval_v2.h5"

# # 需要修改的文本数据
# new_text = "Closing the door"

# # 打开 HDF5 文件（可写模式）
# with h5py.File(h5_file_path, "r+") as f:  # `r+` 允许修改现有数据
#     text_group = f["text_annotations"]  # 定位到文本存储的 group

#     key_name = "door-close-v2_text"  # 目标任务文本
#     if key_name in text_group:
#         del text_group[key_name]  # 先删除旧的数据集
#         text_group.create_dataset(key_name, data=np.string_(new_text))  # 创建新的数据集
#         print(f"✅ '{key_name}' 已更新为: {new_text}")
#     else:
#         print(f"❌ 错误: '{key_name}' 不存在于 HDF5 文件中")

import h5py

file_path = "/scr/yusenluo/RoboCLIP/visualization/decoder_only/all_fail_videos.h5"  # 你的 h5 文件路径

with h5py.File(file_path, "r") as f:
    def print_h5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"📂 Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"📁 Group: {name}")  # 如果是 group，说明它是文件夹

    print("🔍 HDF5 File Structure:")
    f.visititems(print_h5_structure)
