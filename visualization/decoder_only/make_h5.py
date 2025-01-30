import os
import h5py
import numpy as np
import imageio
from tqdm import tqdm
from new_task_annotation_v2 import train_gt_annotation, eval_gt_annotation

def extract_gif_frames(gif_path):
    """ 读取 GIF 文件并提取帧，返回 numpy 数组 """
    gif = imageio.mimread(gif_path)
    return np.array([np.array(frame) for frame in gif], dtype=np.uint8)

def process_and_save(base_path, output_h5_path, subfolder):
    """ 处理文件夹中的 GIF，并保存到 HDF5 """
    
    # 创建 HDF5 文件
    with h5py.File(output_h5_path, "w") as data_file:
        total_num = 0

        # 遍历任务文件夹（Task_id1, Task_id2,...）
        for task_name in tqdm(os.listdir(base_path)):
            task_path = os.path.join(base_path, task_name)
            if not os.path.isdir(task_path):
                continue  # 过滤非文件夹
            
            # 只处理 subfolder ("all_fail" or "close_succ")
            gif_path = os.path.join(task_path, subfolder, "1.gif")
            if not os.path.exists(gif_path):
                continue  # 如果没有 1.gif，则跳过
            
            # 读取 GIF 帧
            frames = extract_gif_frames(gif_path)
            if frames.shape[0] == 0:
                continue  # 过滤空 GIF

            # 进行 224x224 中心裁剪
            frame_list = []
            for img in frames:
                h, w, _ = img.shape
                center_x, center_y = w // 2, h // 2
                img_cropped = img[center_y-112:center_y+112, center_x-112:center_x+112, :]
                frame_list.append(img_cropped)

            # 存储到 HDF5（使用任务名称作为 key）
            img_array = np.array(frame_list, dtype=np.uint8)
            data_file.create_dataset(task_name, data=img_array)
            total_num += 1

        print(f"✅ {output_h5_path}: {total_num} tasks processed.")

        # 存储任务文本
        text_group = data_file.create_group("text_annotations")
        key_list, ann_list = list(train_gt_annotation.keys()), list(train_gt_annotation.values())
        key_list += list(eval_gt_annotation.keys())
        ann_list += list(eval_gt_annotation.values())

        for i, key in enumerate(key_list):
            key_name = key + "_text"
            if key_name in text_group:
                del text_group[key_name]  # 先删除已存在的数据集
            text_group.create_dataset(key_name, data=np.string_(ann_list[i]))  # 存储字符串

if __name__ == "__main__":
    base_path = "/home/jzhang96/RoboCLIPv2/liv_train_final/eval_tasks_v2"  # 你的文件夹路径
    process_and_save(base_path, "all_fail_videos.h5", "all_fail")  # 处理 all_fail
    process_and_save(base_path, "close_succ_videos.h5", "close_succ")  # 处理 close_succ
