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

MAIN_H5 = "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_processed.h5"
H5_MERGING_FROM = "openx_embeddings_langtable.h5"
print(f"Adding to {MAIN_H5} from {H5_MERGING_FROM}")

model, processor, tokenizer = load_model("liv")
model = model.cuda()


# make a set to keep track of the tasks we've seen
total_samples = 0
with h5py.File(MAIN_H5, "a") as f:
    num_keys_before = len(f.keys())
    num_trajs_before = sum([len(f[task].keys()) for task in f.keys()])
    with h5py.File(H5_MERGING_FROM, "r") as f2:
        for task in tqdm(f2.keys()):
            if task in f.keys():
                # get all keys that are numeric
                all_f_keys_for_task = list(f[task].keys())
                all_f_keys_for_task = [
                    int(k) for k in all_f_keys_for_task if k.isnumeric()
                ]
                next_key = max(all_f_keys_for_task) + 1
                for key in f2[task].keys():
                    if key.isnumeric():
                        f[task].create_dataset(
                            str(next_key),
                            data=f2[task][key],
                        )
                        next_key += 1
            else:
                f.create_group(task)
                for key in f2[task].keys():
                    f[task].create_dataset(
                        key,
                        data=f2[task][key],
                    )
    num_keys_after = len(f.keys())
    num_trajs_after = sum([len(f[task].keys()) for task in f.keys()])
    print(f"Added {num_keys_after - num_keys_before} extra tasks")
    print(
        f"Added {num_trajs_after - num_trajs_before} extra trajectories (includes datasets not corresponding to trajs)"
    )