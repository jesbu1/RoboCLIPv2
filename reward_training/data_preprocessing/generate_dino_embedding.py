'''
This file can directly run on non-centercroped videos. 
We have a centercrop video just for visulization. 
'''
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from open_x_processing.clip_utils import dino_load_image, mean_pooling
from transformers import AutoTokenizer, AutoModel
from new_task_annotation_v2 import TRAIN_GT_ANN, EVAL_GT_ANN, GENERATE_TRAIN_ANN
TARGET_PATH = "./"
DINO_BATCH_SIZE = 32
MAX_NUM_FRAMES_PER_EPISODE = 32
TRAIN_VIDEO_PATH = "/home/jzhang96/RoboCLIPv2/reward_training/raw_data/metaworld_centercrop_32_train.h5"
EVAL_VIDEO_PATH = "/home/jzhang96/RoboCLIPv2/reward_training/raw_data/metaworld_centercrop_32_eval.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False)
dinov2_vits14 = dinov2_model.to(device)
minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)


def embedding_images(h5_path, new_h5_path):
    '''
    this function is used to process images to dino embeddings
    '''

    h5_file = h5py.File(h5_path, 'r')
    new_h5_file = h5py.File(new_h5_path, 'w')

    for key in tqdm(h5_file.keys()):
        group = h5_file[key]

        if key not in new_h5_file:
            new_h5_file.create_group(key)

        for idx in list(group.keys()):
            videos = np.asarray(group[idx])

            # # Subsample image to certain num frames. Can skip if already subsampled.
            # indices = np.linspace(
            #     0,
            #     len(videos) - 1,
            #     MAX_NUM_FRAMES_PER_EPISODE,
            #     dtype=int,
            # )
            sampled_images = [videos[i] for i in range(len(videos))]

            with torch.inference_mode():
                # batch it
                episode_images_dino = [
                    dino_load_image(img) for img in sampled_images
                ]
                
                episode_images_dino = [
                    torch.concatenate(
                        episode_images_dino[i : i + DINO_BATCH_SIZE]
                    )
                    for i in range(
                        0, len(episode_images_dino), DINO_BATCH_SIZE
                    )
                ]
                embedding_list = []
                for batch in episode_images_dino:
                    episode_image_embeddings = (
                        dinov2_vits14(batch.to(device))
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    embedding_list.append(episode_image_embeddings)
                episode_image_embeddings = np.concatenate(embedding_list)
            new_h5_file[key].create_dataset(
                idx,
                data=episode_image_embeddings,
            )

        # Add language embedding
        lang_embeddings = list()
        if key in TRAIN_GT_ANN:
            anns = [TRAIN_GT_ANN[key]] + GENERATE_TRAIN_ANN[key]
        else:
            anns = [EVAL_GT_ANN[key]]

        for task in anns:
            encoded_input = minilm_tokenizer(
                [task], padding=False, truncation=True, return_tensors="pt"
            ).to(device)

            model_output = minilm_model(**encoded_input)
            minlm_task_embedding = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .detach()
                .numpy()
            )
            lang_embeddings.append(minlm_task_embedding)
        if len(lang_embeddings) == 1:
            lang_embeddings = np.expand_dims(lang_embeddings[0], axis=0)
        else:
            lang_embeddings = np.concatenate(lang_embeddings, axis=0)
        new_h5_file[key].create_dataset("minilm_lang_embedding", data=lang_embeddings)
    new_h5_file.close()


if __name__ == "__main__":
    
    train_h5_path = os.path.join(TARGET_PATH, "metaworld_embeddings_train.h5")
    eval_h5_path = os.path.join(TARGET_PATH, "metaworld_embeddings_eval.h5")
    embedding_images(TRAIN_VIDEO_PATH, train_h5_path)
    embedding_images(EVAL_VIDEO_PATH, eval_h5_path)
    
    

