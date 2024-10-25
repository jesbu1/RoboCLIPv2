from clip_utils import load_model, embedding_text, embedding_image, compute_similarity
import torch
import cv2 
import numpy as np
from PIL import Image
import h5py
import os
from tqdm import tqdm
import imageio





def main(model_name, video_base_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, tokenizer = load_model(model_name)
    model = model.to(device)
    collect_num = 25
    env_names = os.listdir(video_base_path)
    print(env_names, len(env_names))
    # add data
    data_file = h5py.File("/scr/jzhang96/metaworld_25_for_clip_liv.h5", "a")
    if model_name not in data_file.keys():
        model_group = data_file.create_group(model_name)

    for env_name in env_names:
        if env_name.endswith("-v2"):
            env_path = os.path.join(video_base_path, env_name)
            video_names = os.listdir(env_path)

            env_group = model_group.create_group(env_name)


            for video_name in video_names:
                if video_name.endswith(".gif"):
                    
                    video_path = os.path.join(env_path, video_name)
                    save_name = video_name.split(".")[0].split("_")[-1]
                    frames = imageio.mimread(video_path)
                    frames = [frame[:,:,:3] for frame in frames]

                    embeddings = []
                    for frame in frames:
                        image_embeddings = embedding_image(model, processor, Image.fromarray(frame.astype(np.uint8))).squeeze(0)
                        embeddings.append(image_embeddings.detach().cpu().numpy())
                    embeddings = np.array(embeddings)

                    env_group.create_dataset(save_name, data=embeddings)
    data_file.close()














if __name__ == "__main__":
    video_base_path = "/scr/jzhang96/metaworld_25_for_clip/"
    main("clip", video_base_path)
    main("liv", video_base_path)