import h5py
import torch
# from clip_utils import normalize_embeddings, compute_similarity
import json
import random
import numpy as np
# from clip_utils import load_model, embedding_text, embedding_image, SingleLayerMLP
import matplotlib.pyplot as plt
import wandb
import matplotlib
from tqdm import tqdm
import imageio
# from animation_utils import animate_video_with_rewards, log_gif_to_wandb, compute_mmrv, animate_video_with_rewards_class 
import torch.nn.functional as F
from liv import load_liv
import clip
from PIL import Image
from dataset import normalize_embeddings
import torchvision.transforms as T
from eval_confusion_matrix import padding_video


matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    model = load_liv()
    model.eval()

    return model

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

def real_video_plot(model):
    text = "Put the carrot in the black bowl."
    liv_model = load_liv()
    liv_model.eval()
    text = [text]
    text = clip.tokenize(text)
    text_features = liv_model(input=text, modality="text")
    text_features = normalize_embeddings(text_features)

    liv_model.to("cuda")

    video_path = "/home/jzhang96/RoboCLIPv2/real_robot_exp/jesse_datasets_videos/2024-09-21_11-40-30/traj2.mp4"
    video = imageio.get_reader(video_path)
    frames = []
    for frame in video:
        # save the image as png

        if type(frame) != Image.Image:
            frame = Image.fromarray(frame.astype(np.uint8))
            
            

            image = T.ToTensor()(frame).unsqueeze(0).to("cuda")
            with torch.no_grad():
                image_features = liv_model(input=image, modality="vision")
            image_features = normalize_embeddings(image_features)
            frames.append(image_features)
    frames = torch.cat(frames, dim=0)
    frames = padding_video(frames, 50)
    frames = frames.unsqueeze(0)
    

    triangle_mask = torch.tril(torch.ones(frames.shape[1], frames.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(frames.shape[0], 1, 1, 1)
    mask = None
    pred_class, class_output = model(frames, triangle_mask, text_features, mask = None)


    pred_class = pred_class.squeeze(1)

    predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())

    frame_index = np.linspace(1, len(predicted_classes), len(predicted_classes))

    figure = plt.figure()
    
    plt.plot(frame_index, predicted_classes, label="Correct Text", color="blue")
    plt.xlabel("Frame Index")
    plt.ylabel("Class")
    plt.title("Put the carrot in the black bowl.")
    # if args.catagorical_progress:
    #     plt.ylim(-1, 6)
    # else:
    plt.ylim(-1, 1)
    # plt.savefig(f"progress_img/{env}.png")
    plt.legend()
    # wandb.log("class_test/test real": wandb.Image(figure))
    wandb.log({"class_test/test real": wandb.Image(figure)})
    plt.close()



