from sklearn.decomposition import PCA
import json
import h5py
import numpy as np
from clip_utils import normalize_embeddings
import torch
import torch.nn.functional as F

class SingleLayerMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerMLP, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x
        


def compute_M(X_S, X_T):
    M = np.dot(X_S, X_T.T)  # 35 35
    M_tensor = torch.from_numpy(M).float()
    # if filter:
    #     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/filter"
    # else:
    #     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/pca_matrix_models"
    # os.makedirs(save_dir, exist_ok=True)
    # M_model_path = f"{save_dir}/M_model_{variance_threshold}_Seed{seed}.pth"
    # th.save(M_tensor, M_model_path)
    # print(f'M model saved to {M_model_path}')
    return M_tensor


def pca_learner(h5_file):
    subset_list = json.load(open("task_subset.json"))
    subset_name = "subset_6"
    keys = subset_list[subset_name]
    model_name = "liv"
    model_group = h5_file[model_name]

    text_total_data = list()
    video_total_data = list()
    for key in keys:
        text_key = key + "_text"
        text_data = model_group[text_key][()]
        text_total_data.append(text_data)

        video_group = model_group[key]
        video_keys = list(video_group.keys())[:15]
        videodata = list()
        for video_key in video_keys:
            
            data = video_group[video_key][()]
            video_total_data.append(data)

    text_total_data = np.concatenate(text_total_data, axis = 0)
    video_total_data = np.concatenate(video_total_data, axis = 0)
    text_total_data = normalize_embeddings(text_total_data)
    video_total_data = normalize_embeddings(video_total_data)
    # print("text_total_data shape: ", text_total_data.shape)
    # print("video_total_data shape: ", video_total_data.shape)
    dim = text_total_data.shape[0] - 2 # 148 dimension for 4 head attention
    pca_video = PCA(n_components = dim)
    pca_video.fit(video_total_data)
    pca_text = PCA(n_components = dim)
    pca_text.fit(text_total_data)

    transformation_model = SingleLayerMLP(dim, dim)
    transformation_matrix = compute_M(pca_text.components_, pca_video.components_)
    transformation_model.linear.weight.data = transformation_matrix

    return pca_video, pca_text, transformation_model








if __name__ == "__main__":
    h5_file = h5py.File("/scr/jzhang96/metaworld_25_for_clip_liv.h5", "r")
    pca_learner(h5_file)





        

