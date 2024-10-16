import clip
import torch
import imageio
import numpy as np
from PIL import Image
from liv import load_liv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from sklearn.decomposition import PCA
import json
import os
import joblib







device = "cuda" if torch.cuda.is_available() else "cpu"



def load_model(model_name):
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
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
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

class SingleLayerMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerMLP, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

class TwoLayerMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.linear2 = torch.nn.Linear(input_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.tanh(x)
        return x

class TwoLayerClassMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TwoLayerClassMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.linear2 = torch.nn.Linear(input_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ThreeLayerMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.linear2 = torch.nn.Linear(input_dim, output_dim)
        self.linear3 = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def pca_learner(h5_file, model_name, only_goal_image = True, pca_var = 0.95, experiment_name = None):

    folder_path = "pca_models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



    # text_file_name = f"{folder_path}/{model_name}_text_pca_model_var" + str(pca_var)
    # image_file_name = f"{folder_path}/{model_name}_image_pca_model_var" + str(pca_var)

    # if only_goal_image:
    #     text_file_name = text_file_name + "_goal"
    #     image_file_name = image_file_name + "_goal"

    # text_file_name = text_file_name + ".pkl"
    # image_file_name = image_file_name + ".pkl"

    text_file_name = f"{folder_path}/{experiment_name}.pkl"
    image_file_name = f"{folder_path}/{experiment_name}.pkl"


    if os.path.exists(text_file_name) and os.path.exists(image_file_name):
        text_pca_model = joblib.load(text_file_name)
        image_pca_model = joblib.load(image_file_name)

    else:
        train_envs = json.load(open("task_subset.json"))["subset_6"]
        text_embeddings = []
        image_embeddings = []
        for env in train_envs:
            text_env_name = f"{env}_text"
            text_array = h5_file[model_name][text_env_name][:]
            text_embeddings.append(np.array(text_array))

            image_dataset = h5_file[model_name][env]

            for key in sorted(image_dataset.keys(), key = int)[:15]:
                image_array = image_dataset[key][:]
                if only_goal_image:
                    image_array = image_array[-1:]
                image_embeddings.append(np.array(image_array))

        text_embeddings = np.concatenate(text_embeddings, axis=0)
        image_embeddings = np.concatenate(image_embeddings, axis=0)
        text_embeddings = normalize_embeddings(torch.from_numpy(text_embeddings).to(device))
        image_embeddings = normalize_embeddings(torch.from_numpy(image_embeddings).to(device))
        print("text_embeddings shape", text_embeddings.shape)
        print("image_embeddings shape", image_embeddings.shape)
        
        if pca_var < 1:
            text_pca_model = PCA(n_components = pca_var)
        else:
            text_pca_model = PCA(n_components = text_embeddings.shape[0])
        text_pca_model.fit(text_embeddings.cpu())
        print("Text PCA Model Fitted", text_pca_model.n_components_)
        joblib.dump(text_pca_model, text_file_name)
        
        

        image_pca_model = PCA(n_components = text_pca_model.n_components_)
        image_pca_model.fit(image_embeddings.cpu())
        print("Image PCA Model Fitted", image_pca_model.n_components_)
        joblib.dump(image_pca_model, image_file_name)

    return text_pca_model, image_pca_model


def compute_M(X_S, X_T):
    M = np.dot(X_S, X_T.T)  # 35 35
    M_tensor = torch.from_numpy(M).float()

    return M_tensor










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

    