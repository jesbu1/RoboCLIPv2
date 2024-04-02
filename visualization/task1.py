import PIL
from s3dg import S3D
import torch as th
import json
#from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2
from pca import plot_embeddings_3d,plot_embeddings
from mlp import mlp_eval
from torch.utils.data import Dataset, DataLoader
from Dataload import VideoTextDataset, Embedding, list_webm_files
import joblib





if __name__ == '__main__':
    if th.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    video_paths = list_webm_files('../20bn-something-something-v2/validation') #'../20bn-something-something-v2'
    #print(video_paths)
    video_text_dataset = VideoTextDataset(video_paths, num_samples = 50, random_samples= False, dataset_type='validation')
    data_loader = DataLoader(video_text_dataset, batch_size=25, shuffle=True, num_workers=2)

    s3d = S3D('../s3d_dict.npy', 512)
    s3d.load_state_dict(th.load('../s3d_howto100m.pth'))
    s3d.eval()
    video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(s3d, data_loader)
    #print(video_embeddings.shape, text_embeddings.shape)
    l2_distances = th.norm(text_embeddings - video_embeddings, p=2, dim=1)
    similarity_scores = th.matmul(text_embeddings, video_embeddings.t())
    # video_variances = th.var(video_embeddings, dim=0)
    # text_variances = th.var(text_embeddings, dim=0)
    # mean_video_variance = th.mean(video_variances)
    # mean_text_variance = th.mean(text_variances)
    #
    # print(f"Mean variance in video embeddings: {mean_video_variance}")
    # print(f"Mean variance in text embeddings: {mean_text_variance}")

    mean_distance = th.mean(l2_distances)
    std_distance = th.std(l2_distances)
    min_distance = th.min(l2_distances)
    max_distance = th.max(l2_distances)

    mean_score = th.mean(similarity_scores)
    min_score = th.min(similarity_scores)
    max_score = th.max(similarity_scores)
    std_score = th.std(similarity_scores)

    print("Mean similarity score:", mean_score.item())
    print("Min similarity score:", min_score.item())
    print("Max similarity score:", max_score.item())
    print("STD of similarity scores:", std_score.item())
    print("Mean L2 Distance:", mean_distance.item())
    print("STD L2 Distance:", std_distance.item())
    print("Min L2 Distance:", min_distance.item())
    print("Max L2 Distance:", max_distance.item())

    variance_thresholds = [0.9, 0.95, 0.99]
    sample_sizes = [1, 2, 4, 8, 16]
    for variance_threshold in variance_thresholds:
        for sample_size in sample_sizes:
            sample_size = 50 * sample_size
            pca_video_model_path = f'saved_model/pca_model_video_{variance_threshold}_{sample_size}.pkl'
            video_pca = joblib.load(pca_video_model_path)
            pca_text_model_path = f'saved_model/pca_model_text_{variance_threshold}_{sample_size}.pkl'
            text_pca = joblib.load(pca_text_model_path)


            #print(video_embeddings_pca.shape[1], text_embeddings_pca.shape[1])
            print(f"Results with variance_threshold {variance_threshold}:")

            print("Result before MLP")
            plot_embeddings(video_embeddings, text_embeddings, mappings,
                            'plots', f'pca_plot_nomlp2D_{variance_threshold}_{sample_size}.png', False)
            plot_embeddings_3d(video_embeddings, text_embeddings, mappings,
                               'plots', f'pca_plot_nomlp3D_{variance_threshold}_{sample_size}.png', False)

            video_embeddings_pca = video_pca.transform(video_embeddings)
            text_embeddings_pca = text_pca.transform(text_embeddings)
            # 将PCA处理后的嵌入转换回torch.Tensor，并进行MLP评估
            video_embeddings_tensor = th.from_numpy(video_embeddings_pca).float().to(device)
            text_embeddings_tensor = th.from_numpy(text_embeddings_pca).float().to(device)

            # 注意确保mlp_eval函数和模型在正确的设备上执行
            mlp_model_path = f'saved_model/final_model_{variance_threshold}_{sample_size}.pth'
            adjusted_video_embeddings = mlp_eval(video_embeddings_tensor, text_embeddings_tensor, mlp_model_path)

            print("Result after MLP")
            plot_embeddings(adjusted_video_embeddings.cpu().numpy(), text_embeddings_pca, mappings,
                            'plots', f'pca_plot_mlp2D_{variance_threshold}_{sample_size}.png', False)
            plot_embeddings_3d(adjusted_video_embeddings.cpu().numpy(), text_embeddings_pca, mappings,
                               'plots', f'pca_plot_mlp3D_{variance_threshold}_{sample_size}.png', False)

