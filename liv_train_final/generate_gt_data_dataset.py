from clip_utils import load_model, embedding_text, embedding_image, compute_similarity
import torch
import cv2 
import numpy as np
from PIL import Image
import h5py
import os
from tqdm import tqdm
import imageio
import json
from new_task_annotation_v2 import train_gt_annotation, eval_gt_annotation





def main(model_name, video_base_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, tokenizer = load_model(model_name)
    model = model.to(device)
    # collect_num = 25
    # add data
    video_path = "/scr/jzhang96/metaworld_traj_15_demos.h5"
    video_file = h5py.File(video_path, "r")
    metaworld_embedding_dataset = "metaworld_embedding_1_demo_dataset_v2.h5"
    data_file = h5py.File(metaworld_embedding_dataset, "w")
    total_num = 0

    
    for key_name in tqdm(video_file.keys()):
        video_dataset = video_file[key_name]
        env_id = video_dataset["env_id"][()][-1].decode("utf-8")

        if env_id in data_file.keys():
            continue
        else:
            imgs = video_dataset["img"][:]
            frame_list = []
            for i in range(imgs.shape[0]):
                img = imgs[i]
                # center crop 224x224
                img = img[240-112:240+112, 320-112:320+112, :]
                image_embeddings = embedding_image(model, processor, Image.fromarray(img.astype(np.uint8))).squeeze().detach().cpu().numpy()
                frame_list.append(image_embeddings)
            image_embeddings = np.array(frame_list)
            data_file.create_dataset(env_id, data=image_embeddings)
            total_num += 1

    print(total_num)


    key_list = []
    ann_list = []

    for key in train_gt_annotation:
        key_list.append(key)
        ann_list.append(train_gt_annotation[key])

    for key in eval_gt_annotation:
        key_list.append(key)
        ann_list.append(eval_gt_annotation[key])

    key_list += ["clean-table"]
    ann_list += ["Cleaning the table"]



    annotation_embeddings = embedding_text(model, tokenizer, ann_list).detach().cpu().numpy()
    for i, key in enumerate(key_list):
        key_name = key + "_text"
        data_file.create_dataset(key_name, data=annotation_embeddings[i])

    data_file.close()


        















if __name__ == "__main__":
    video_base_path = "/scr/jzhang96/metaworld_25_for_clip/"
    # main("clip", video_base_path)
    main("liv", video_base_path)