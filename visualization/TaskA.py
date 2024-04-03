import PIL
from s3dg import S3D
import torch as th
import json

# from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2
from pca import plot_embeddings_3d, plot_embeddings, check_pairs
from mlp import mlp_eval, normalize_embeddings
from torch.utils.data import Dataset, DataLoader
from Dataload import VideoTextDataset, Embedding, list_webm_files
import joblib
from sklearn.decomposition import PCA

START_SAMPLE_SIZE = 50
VAL_SAMPLE_SIZE = 150

if __name__ == "__main__":
    if th.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    validation_video_paths = list_webm_files(
        "../20bn-something-something-v2/validation"
    )  #'../20bn-something-something-v2'
    training_video_paths = list_webm_files(
        "../20bn-something-something-v2/train"
    )  # '../20bn-something-something-v2'
    # print(video_paths)
    training_dataset = VideoTextDataset(
        training_video_paths,
        num_samples=800,
        random_samples=False,
        dataset_type="train",
    )
    validation_dataset = VideoTextDataset(
        validation_video_paths,
        num_samples=150,
        random_samples=False,
        dataset_type="validation",
    )
    # data_loader = DataLoader(
    #     training_dataset, batch_size=25, shuffle=True, num_workers=2
    # )

    data_loader = DataLoader(
        validation_dataset, batch_size=25, shuffle=True, num_workers=2
    )

    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(th.load("../s3d_howto100m.pth"))
    s3d.eval()
    """
        这里是 没有normalized的 Embeddings
    """
    video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(
        s3d, data_loader
    )

    similarity_scores = th.matmul(text_embeddings, video_embeddings.t())

    mean_score = th.mean(similarity_scores)
    min_score = th.min(similarity_scores)
    max_score = th.max(similarity_scores)
    std_score = th.std(similarity_scores)
    print("Summary of UNnormalized embeddings")
    print("Mean similarity score before norm:", mean_score.item())
    print("Min similarity score before norm:", min_score.item())
    print("Max similarity score before norm:", max_score.item())
    print("STD of similarity scores before norm:", std_score.item())

    """
    这里是 normalized_Embeddings
    """

    video_embeddings_normalized = normalize_embeddings(video_embeddings).clone()
    text_embeddings_normalized = normalize_embeddings(text_embeddings).clone()

    similarity_scores_normalized = th.matmul(
        text_embeddings_normalized, video_embeddings_normalized.t()
    )

    mean_score_normalized = th.mean(similarity_scores_normalized)
    min_score_normalized = th.min(similarity_scores_normalized)
    max_score_normalized = th.max(similarity_scores_normalized)
    std_score_normalized = th.std(similarity_scores_normalized)
    print("Summary of normalized embeddings")
    print("Mean similarity score after norm:", mean_score_normalized.item())
    print("Min similarity score after norm:", min_score_normalized.item())
    print("Max similarity score after norm:", max_score_normalized.item())
    print("STD of similarity scores after norm:", std_score_normalized.item())

    print(
        f"TOP K accuracies of normalized embeddings without PCA"
    )
    check_pairs(video_embeddings.numpy(), text_embeddings.numpy(), mappings, False)
    print(
        "TOP K accuracies of UNnormalized embeddings without PCA"
    )
    check_pairs(
        video_embeddings_normalized.numpy(),
        text_embeddings_normalized.numpy(),
        mappings,
        False,
    )
    """
        这里是 使用2d PCA的
    """
    combined_embeddings = np.vstack((video_embeddings, text_embeddings))
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    combined_embeddings_norm = np.vstack((video_embeddings_normalized, text_embeddings_normalized))
    pca_norm = PCA(n_components=2)
    reduced_embeddings_norm = pca_norm.fit_transform(combined_embeddings_norm)

    reduced_video_embeddings = th.from_numpy(reduced_embeddings[: len(video_embeddings)]).float()
    reduced_text_embeddings = th.from_numpy(reduced_embeddings[len(text_embeddings):]).float()

    reduced_video_embeddings_norm = th.from_numpy(reduced_embeddings_norm[: len(video_embeddings_normalized)]).float()
    reduced_text_embeddings_norm = th.from_numpy(reduced_embeddings_norm[len(text_embeddings_normalized):]).float()

    similarity_scores_PCA = th.matmul(
        reduced_text_embeddings, reduced_video_embeddings.t()
    )

    similarity_scores_PCA_norm = th.matmul(
        reduced_text_embeddings_norm, reduced_video_embeddings_norm.t()
    )

    print(
        f"TOP K accuracies of normalized embeddings with 2D PCA"
    )
    check_pairs(reduced_video_embeddings_norm.numpy(), reduced_text_embeddings_norm.numpy(), mappings, False)
    plot_embeddings(reduced_video_embeddings_norm,
                    reduced_text_embeddings_norm, mappings, 'plots/taskA', 'PCA_2D_norm', False)
    print(
        "TOP K accuracies of UNnormalized embeddings with 2D PCA"
    )
    check_pairs(
        reduced_video_embeddings.numpy(),
        reduced_text_embeddings.numpy(),
        mappings,
        False,
    )
    plot_embeddings(reduced_video_embeddings,
                    reduced_text_embeddings, mappings, 'plots/taskA', 'PCA_2D_Unnorm', False)


