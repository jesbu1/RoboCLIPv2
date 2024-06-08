'''
This file is used to preprocess dataset for x-clip model, and save to h5 files for fast data loading
'''
import os
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoProcessor 
from xclip_utils import readGif, preprocess_human_demo_xclip, adjust_frames_xclip, load_xclip_model
import torch
from pca_utils import plot_embeddings, normalize_embeddings

def generate_droid_pair_data_to_h5(dataset_name = 'droid',
                                   video_folder_path = '/scr/yusenluo/RoboCLIP/OpenX/droid',
                                   csv_path = '/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv',
                                   model_name = 'microsoft/xclip-base-patch16-zero-shot',
                                    ):
    '''
    generate the video text pair data to h5 file
    '''
    # load the csv file
    df = pd.read_csv(csv_path)
    processor = AutoProcessor.from_pretrained(model_name)
    length = len(df)

    h5_file = h5py.File("/scr/jzhang96/droid_sth_dataset.hdf5", "w")

    for i in range (length):
        row = df.iloc[i]
        video_id = row[dataset_name].replace(' ', '_')
        text_label = row[dataset_name]

        video_path = os.path.join(video_folder_path, f"{video_id}.gif")
        frames = readGif(video_path)
        frames = frames[:, :, :, :3] # remove alpha channel
        
        frames = preprocess_human_demo_xclip(frames)
        frames = adjust_frames_xclip(frames)

        frames = [frames[i] for i in range (frames.shape[0])]
        processor_output = processor(videos=frames, return_tensors="pt")
        
        video_pixel_values = processor_output["pixel_values"].squeeze(0).cpu().detach().numpy()

        new_group = h5_file.create_group(str(i))
        new_group.attrs["text"] = text_label
        new_group.attrs["video_id"] = video_id
        new_group.create_dataset("frames", data=video_pixel_values)

        # to get frames
        # video_pixel_values = new_group["frames"][:]
        # text = new_group.attrs["text"]

def get_droid_text_video_latent(h5_path = "/scr/jzhang96/droid_sth_dataset.hdf5",
                                model_name = 'microsoft/xclip-base-patch16-zero-shot',):
    '''
    get the latent features of the video and text
    '''
    h5_file = h5py.File(h5_path, "r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xclip_tokenizer, xclip_processor, xclip_model = load_xclip_model(model_name, device)
    video_feature_total = None
    text_feature_total = None
    new_h5_file = h5py.File("/scr/jzhang96/droid_sth_dataset_latent.hdf5", "w")

    for i in tqdm(h5_file.keys()):
        new_h5_file.create_group(i)
        item = h5_file[str(i)]
        video_frames = item["frames"][:]
        video_frames = torch.tensor(video_frames).to(device)
        video_features = xclip_model.get_video_features(video_frames.unsqueeze(0))
        text = item.attrs["text"]
        text_tokens = xclip_tokenizer([text], padding=True, return_tensors="pt").to(device)
        text_features = xclip_model.get_text_features(**text_tokens)

        video_features = video_features.cpu().detach().numpy()
        text_features = text_features.cpu().detach().numpy()
        norm_video_features = video_features / np.linalg.norm(video_features, axis=1, keepdims=True)
        norm_text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        video_features = np.squeeze(video_features)
        text_features = np.squeeze(text_features)
        norm_video_features = np.squeeze(norm_video_features)
        norm_text_features = np.squeeze(norm_text_features)

        new_h5_file[i].create_dataset("video_features", data=video_features)
        new_h5_file[i].create_dataset("text_features", data=text_features)
        new_h5_file[i].create_dataset("video_norm_features", data=norm_video_features)
        new_h5_file[i].create_dataset("text_norm_features", data=norm_text_features)
        new_h5_file[i].attrs["text"] = text
        # new_h5_file[i].create_dataset("video_norm_feature", data=
        # norm_features = 

        # if video_feature_total is None:
        #     video_feature_total = video_features.cpu().detach().numpy()
        #     text_feature_total = text_features.cpu().detach().numpy()
        # else:
        #     video_feature_total = np.concatenate((video_feature_total, video_features.cpu().detach().numpy()), axis=0)
        #     text_feature_total = np.concatenate((text_feature_total, text_features.cpu().detach().numpy()), axis=0)

    # video_feature_total = normalize_embeddings(video_feature_total)
    # text_feature_total = normalize_embeddings(text_feature_total)

    # for i in tqdm(new_h5_file.keys()):
    #     new_h5_file[i].create_dataset("video_norm_features", data=video_feature_total[int(i)])
    #     new_h5_file[i].create_dataset("text_norm_features", data=text_feature_total[int(i)])
    # new_h5_file.create_dataset("video_feature_total", data=video_feature_total)
    # new_h5_file.create_dataset("text_feature_total", data=text_feature_total)
    new_h5_file.close()



get_droid_text_video_latent()


# generate_droid_pair_data_to_h5()                                   