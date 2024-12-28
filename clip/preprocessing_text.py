from clip_utils import load_model, embedding_text, embedding_image, compute_similarity
import torch
import cv2 
import numpy as np
from PIL import Image
import h5py
import os
from tqdm import tqdm
import imageio
from generated_text import generate_set_6_ann_v2, gt_annotations_v2




def main(model_name, generated_text, gt_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, tokenizer = load_model(model_name)
    model = model.to(device)



    # add data
    # data_file = h5py.File("/scr/jzhang96/metaworld_25_for_clip_liv.h5", "a")

    # if model_name not in data_file.keys():
    #     model_group = data_file.create_group(model_name)
    # model_group = data_file[model_name]
    for env_name in tqdm(generated_text.keys()):
        text_env_name = env_name + "_text"

        gt_ann = gt_text[env_name]
        generated_ann = generated_text[env_name]
        anns = [gt_ann] + generated_ann[:-1]

        # env_text_group = model_group.create_dataset(text_env_name, data=anns)
        text_embeddings = embedding_text(model, tokenizer, anns)
        text_embeddings = text_embeddings.detach().cpu().numpy()
        norm = np.linalg.norm(text_embeddings, axis=1)
        print(norm)
        import pdb; pdb.set_trace()
        # model_group.create_dataset(text_env_name, data=text_embeddings)


    data_file.close()

    # for env_name in env_names:
    #     if env_name.endswith("-v2"):
    #         env_path = os.path.join(video_base_path, env_name)
    #         video_names = os.listdir(env_path)

    #         env_group = model_group.create_group(env_name)


    #         for video_name in video_names:
    #             if video_name.endswith(".gif"):
                    
    #                 video_path = os.path.join(env_path, video_name)
    #                 save_name = video_name.split(".")[0].split("_")[-1]
    #                 frames = imageio.mimread(video_path)
    #                 frames = [frame[:,:,:3] for frame in frames]

    #                 embeddings = []
    #                 for frame in frames:
    #                     image_embeddings = embedding_image(model, processor, Image.fromarray(frame.astype(np.uint8))).squeeze(0)
    #                     embeddings.append(image_embeddings.detach().cpu().numpy())
    #                 embeddings = np.array(embeddings)

    #                 env_group.create_dataset(save_name, data=embeddings)
    # data_file.close()









if __name__ == "__main__":
    # main("clip", generate_set_6_ann_v2, gt_annotations_v2)
    main("liv", generate_set_6_ann_v2, gt_annotations_v2)
