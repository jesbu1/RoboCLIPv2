import h5py
from tqdm import tqdm
import numpy as np

from clip_utils import (
    load_model,
    embedding_text,
    embedding_image,
    get_full_liv_embedding,
    dino_load_image,
    mean_pooling,
)
import torch
import imageio
from transformers import AutoTokenizer, AutoModel

DINO_BATCH_SIZE = 32
MAX_NUM_FRAMES_PER_EPISODE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dinov2_vits14 = dinov2_vits14.to(device)
minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)

path = "/home/jzhang96/metaworld_traj_15_demos.h5"
h5_file = h5py.File(path, 'r')

new_file_name = "metaworld_dino_embeddings.h5"
new_h5_file = h5py.File(new_file_name, 'w')

for key in tqdm(h5_file.keys()):
    group = h5_file[key]
    env_id = group["env_id"][()][0].decode("utf-8")

    if env_id not in new_h5_file:
        new_h5_file.create_group(env_id)
    videos = np.asarray(group['img'])

    indices = np.linspace(
        0,
        len(videos) - 1,
        MAX_NUM_FRAMES_PER_EPISODE,
        dtype=int,
    )
    sampled_images = [videos[i] for i in indices]


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

        length = len(new_h5_file[env_id])
        dataset_name = str(length)
        new_h5_file[env_id].create_dataset(
            dataset_name,
            data=episode_image_embeddings,
        )
print("done", len(new_h5_file.keys()))
new_h5_file.close()



    # save to png

