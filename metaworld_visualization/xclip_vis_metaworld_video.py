import os
import sys
# add the "../" directory to the sys.path
parent_dir_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
parent_dir_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xclip/'))
sys.path.append(parent_dir_1)
sys.path.append(parent_dir_2)
import cv2
import random
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from tqdm import tqdm
# import xclip
from xclip_utils import preprocess_human_demo_xclip, adjust_frames_xclip
from pca_utils import normalize_embeddings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import AutoTokenizer, AutoModel, AutoProcessor 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def get_models(model_name):
    '''
    Get the model and tokenizer.
    '''
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, tokenizer, processor

def video2array(video_path): # 
    '''
    Convert a video to an array (rgb) of frames.
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def sample_videos(video_folder, num_samples=10):
    '''
    Sample num_samples videos from the video_folder.
    '''
    all_files = os.listdir(video_folder)
    sampled_files = random.sample(all_files, num_samples)
    output_list = []
    for file in sampled_files:
        file_path = os.path.join(video_folder, file)
        frames = video2array(file_path)
        output_list.append((file_path, frames))
    return output_list

def check_folder(folder_path, num_videos=30):
    '''
    Check if folder have enough videos.
    '''
    all_files = os.listdir(folder_path)
    if len(all_files) < num_videos:
        return False
    return True

def main():
    '''
    Main function.
    '''
    model_name = "microsoft/xclip-base-patch16-zero-shot"
    model, tokenizer, processor = get_models(model_name)
    model.cuda()
    total_video_features = []
    true_folders = 0
    num = 0
    text = []
    for i, key in tqdm(enumerate(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())):
        folder_name = "videos_" + key
        prefix = "/scr/jzhang96/metaworld_generation_true_video/" + folder_name
        if check_folder(prefix):
            videos = sample_videos(prefix)
            true_folders += 1
            text.append(key.split("-v2")[0])
            for video in videos:
                new_video = preprocess_human_demo_xclip(video[1])
                new_video = adjust_frames_xclip(new_video)
                new_video = [new_video[i] for i in range (new_video.shape[0])]
                new_video = processor(videos=new_video, return_tensors="pt")
                new_video = new_video["pixel_values"].cuda()
                video_features = model.get_video_features(new_video).squeeze(0).cpu().detach().numpy()
                total_video_features.append(video_features)
                num += 1

    total_video_features = np.array(total_video_features)
    total_video_features = normalize_embeddings(total_video_features)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(total_video_features)


    # Plot the reduced embeddings
    colors = plt.cm.tab10(np.linspace(0, 1, true_folders))  # 28 groups

    for i in range(true_folders):
        group_data = reduced_embeddings[i*10:(i+1)*10]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = text[i]
        plt.scatter(x, y, color=colors[i], label=text_name)
    
    plt.title('2D PCA for Metaworld Success Videos')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')

    plt.legend(loc='upper left', ncol=4)
    plt.tight_layout(rect=[0, 0, 0.5, 1]) # adjust the plot to the right (to fit the legend

    # plt.legend()
    # save png
    plt.savefig('pca_metaworld_success_videos.png')


    # print("Total number of tasks with enough videos: ", num)

if __name__ == "__main__":
    main()