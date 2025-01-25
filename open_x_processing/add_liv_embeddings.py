import tensorflow_datasets as tfds
import random
from tqdm import tqdm
import os
from oxe_configs import OXE_DATASET_CONFIGS
import json
import numpy as np
import h5py
from clip_utils import (
    load_model,
    embedding_text,
    embedding_image,
    get_full_liv_embedding,
)
from PIL import Image

LOAD_H5_NAME = "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable35k.h5" #"/data/shared/roboclip/openx_embeddings_full_uncompressed.h5"  # TODO: grab the right one
print(f"Adding to {LOAD_H5_NAME}")
DEBUG = False  # will only make 10 per dataset

model, processor, tokenizer = load_model("liv")
model = model.cuda()


# make a set to keep track of the tasks we've seen
total_samples = 0
with h5py.File(LOAD_H5_NAME, 'a') as f:
    for task in tqdm(f.keys()):
        task_embedding = (
            get_full_liv_embedding(model, tokenizer, [task]).detach().cpu().numpy()
        )
        # create a dataset with the embeddings
        if "lang_embedding_individual" not in f[task].keys():
            f[task].create_dataset("lang_embedding_individual", data=task_embedding)
