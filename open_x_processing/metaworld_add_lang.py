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
from new_task_annotation_v2 import train_gt_annotation, eval_gt_annotation, generated_gt_annotation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)

h5_file_name = "metaworld_dino_embeddings.h5"
h5_file = h5py.File(h5_file_name, 'a')


for key in train_gt_annotation.keys():
    gt_ann = train_gt_annotation[key]
    extend_ann = generated_gt_annotation[key]
    ann = [gt_ann] + extend_ann


    lang_embeddings = list()

    for task in ann:
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
    lang_embeddings = np.concatenate(lang_embeddings, axis=0)

    if key in h5_file:
        if "minilm_lang_embedding" in h5_file[key]:
            del h5_file[key]["minilm_lang_embedding"]
        print(f"Adding Key {key}")
        h5_file[key].create_dataset("minilm_lang_embedding", data=lang_embeddings)
        
    else:
        print(f"Train Key {key} not found in h5 file")

for key in eval_gt_annotation.keys():
    gt_ann = eval_gt_annotation[key]
    ann = [gt_ann]


    lang_embeddings = list()

    for task in ann:
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
    lang_embeddings = np.concatenate(lang_embeddings, axis=0)
    if key in h5_file:
        if "minilm_lang_embedding" in h5_file[key]:
            print(h5_file[key]["minilm_lang_embedding"].shape,key)
            del h5_file[key]["minilm_lang_embedding"]
        h5_file[key].create_dataset("minilm_lang_embedding", data=lang_embeddings)
    else:
        print(f"Eval Key {key} not found in h5 file ")


                          
