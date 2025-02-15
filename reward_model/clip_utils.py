import clip
import torch
import imageio
import numpy as np
from PIL import Image
from liv import load_liv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T

# from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from sklearn.decomposition import PCA
import json
import os
import joblib


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name="liv"):
    if model_name == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    elif model_name == "liv":
        model = load_liv()
        model.eval()
        processor = None
        tokenizer = None
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, processor, tokenizer


def get_full_liv_embedding(model, tokenizer, text):
    if type(text) != list:
        text = [text]
    text_tokens = clip.tokenize(text).to(model.module.device)

    def encode_text(model, text):
        x = model.token_embedding(text).type(
            model.dtype
        )  # [batch_size, n_ctx, d_model]
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x).type(model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ model.text_projection
        return x

    text_embeddings = encode_text(model.module.model, text_tokens)
    # truncate it
    zero_indices = torch.nonzero(text_tokens == 0)
    text_embeddings = text_embeddings[:, : zero_indices[0][1]]

    return text_embeddings


def embedding_text(model, tokenizer, text):
    if type(text) != list:
        text = [text]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is not None:
        inputs = tokenizer(text=text, return_tensors="pt", padding=True)
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeddings = model.get_text_features(**inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    else:
        text = clip.tokenize(text)
        text_embeddings = model(input=text, modality="text")
    return text_embeddings


def embedding_image(model, processor, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(image) != Image.Image:
        image = Image.fromarray(image.astype(np.uint8))
    if processor is not None:
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
    else:
        image = T.ToTensor()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embeddings = model(input=image, modality="vision")
    return image_embeddings


def compute_similarity(text_embeddings, image_embeddings):
    cosine_similarity = F.cosine_similarity(text_embeddings, image_embeddings)
    return cosine_similarity


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


if __name__ == "__main__":
    model_name = "clip"
    # model_name = "liv"
    model, processor, tokenizer = load_model(model_name)
    text = "Robot closing a door"
    text_embeddings = embedding_text(model, tokenizer, text)

    image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    image = Image.fromarray(image)
    image_embeddings = embedding_image(model, processor, image)

    similarity = compute_similarity(text_embeddings, image_embeddings)
    print(similarity)
    # hardnesses = ["all_fail", "close_fail", "success", "GT"]
