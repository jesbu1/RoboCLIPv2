import h5py
import numpy as np
import os
import cv2
from liv import load_liv
import torch
from PIL import Image
import clip
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

def normalize_embeddings(embeddings, return_tensor=False):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


def load_text_annotations(file_path):
    with open(file_path, 'r') as file:
        single_line = file.readline().strip() 
    return single_line



def load_video_to_frames(mp4_file):
    frames = []  # List to store frames as arrays

    # Open the video file
    cap = cv2.VideoCapture(mp4_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when no frames are left

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)  # Append the RGB frame to the list

    cap.release()
    return frames


def load_model():

    model = load_liv()
    model.eval()

    return model

def embedding_text(model, text):
    if type(text) != list:
        text = [text]

    text = clip.tokenize(text)
    text_embeddings = model(input=text, modality="text")
    return text_embeddings


def embedding_image(model, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = T.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = model(input=image, modality="vision")
    return image_embeddings



if __name__ == "__main__":
    video_base_dir = '/scr/jzhang96/jesse_datasets_videos'
    video_classes = os.listdir(video_base_dir)
    text_base_dir = '/scr/jzhang96/jesse_datasets'
    text_base_end = 'raw/traj_group0/traj0/lang.txt'
    liv_model = load_model()
    h5_file = h5py.File('real_video_embeddings_no_norm.h5', 'w')
    text_embeddings_group = h5_file.create_group('text_embeddings')
    video_embedding_group = h5_file.create_group('video_embeddings')
    num = 0
    for video_class in video_classes:
        text_dir = os.path.join(text_base_dir, video_class, text_base_end)
        text_annotations = load_text_annotations(text_dir)

        print(text_annotations)
        video_file_path = os.path.join(video_base_dir, video_class)
        video_file_list = os.listdir(video_file_path)

        text_embedding = embedding_text(liv_model, text_annotations)
        # text_embedding = normalize_embeddings(text_embedding, return_tensor=True)
        dataset = text_embeddings_group.create_dataset(video_class, data=text_embedding.detach().cpu().numpy())
        dataset.attrs['text'] = text_annotations

        video_group = video_embedding_group.create_group(video_class)
        video_group.attrs['text'] = text_annotations

        for video_file in video_file_list:
            video_file_path = os.path.join(video_base_dir, video_class, video_file)
            video = load_video_to_frames(video_file_path)
            video_embeddings = []
            
            for frame in tqdm(video):
                image_embedding = embedding_image(liv_model, frame)
                # image_embedding = normalize_embeddings(image_embedding, return_tensor=True)
                video_embeddings.append(image_embedding)
            video_embeddings = torch.stack(video_embeddings)
            video_embeddings = video_embeddings.detach().cpu().numpy()
            dataset = video_group.create_dataset(video_file, data=video_embeddings)
            num += 1
            print(num)
    h5_file.close()


        


