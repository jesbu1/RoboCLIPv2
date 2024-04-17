import PIL
from s3dg import S3D
import torch as th
import json
import csv
# from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2
from pca import plot_embeddings_3d, plot_embeddings, check_pairs
from mlp import mlp_eval, normalize_embeddings, standradize_embeddings
from torch.utils.data import Dataset, DataLoader
from Dataload import VideoTextDataset, Embedding, list_webm_files
import joblib
from sklearn.decomposition import PCA
import argparse

START_SAMPLE_SIZE = 50
VAL_SAMPLE_SIZE = 150

if __name__ == "__main__":
    if th.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    csv_file_path = "Original_Embeddings_Result.csv"
    if not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0:
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Sample Category", "Sample Size for Test",
                 "Model Name", "Top 1 Accuracy", "Top 3 Accuracy", "Top 5 Accuracy", "Top 10 Accuracy",
                 "Mean Similarity Score", "Min Similarity Score", "Max Similarity Score", "Std Similarity Score"])
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--validation", type=bool, default=False)
    # args = parser.parse_args()
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # validation_video_paths = list_webm_files(
    #     "../20bn-something-something-v2/validation"
    # )  #'../20bn-something-something-v2'
    # training_video_paths = list_webm_files(
    #     "../20bn-something-something-v2/train"
    # )  # '../20bn-something-something-v2'
    # # print(video_paths)
    # training_dataset = VideoTextDataset(
    #     training_video_paths,
    #     num_samples=1050,
    #     random_samples=False,
    #     dataset_type="train",
    # )
    # validation_dataset = VideoTextDataset(
    #     validation_video_paths,
    #     num_samples=150,
    #     random_samples=False,
    #     dataset_type="validation",
    # )
    training_video_paths = list_webm_files(
        "../OpenX/droid/left_1"
    )  # '../20bn-something-something-v2'
    # print(video_paths)
    training_dataset = VideoTextDataset(
        training_video_paths,
        random_samples=False,
        dataset_type="train",
        dataset= 'OpenX',
    )
    data_loader = DataLoader(
                training_dataset, batch_size=50, shuffle=False, num_workers=5
    )
    data_type = "OpenX"
    sample_size = len(training_dataset)
    # if not args.validation:
    #     data_loader = DataLoader(
    #         training_dataset, batch_size=50, shuffle=False, num_workers=5
    #     )
    #     sample_size = 1050
    #     data_type = "Train"
    # else:
    #     data_loader = DataLoader(
    #         validation_dataset, batch_size=50, shuffle=False, num_workers=5
    #     )
    #     sample_size = 150
    #     data_type = "Validation"

    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(th.load("../s3d_howto100m.pth"))
    s3d.eval()
    video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(
        s3d, data_loader
    )
    # """
    #     UNnormalized的 Embeddings
    # """
    similarity_scores = th.matmul(video_embeddings, text_embeddings.t())
    mean_score = (th.mean(similarity_scores)).numpy()
    min_score = (th.min(similarity_scores)).numpy()
    max_score = (th.max(similarity_scores)).numpy()
    std_score = (th.std(similarity_scores)).numpy()
    # print("Summary of UNnormalized embeddings")
    # print("Mean similarity score before norm:", mean_score.item())
    # print("Min similarity score before norm:", min_score.item())
    # print("Max similarity score before norm:", max_score.item())
    # print("STD of similarity scores before norm:", std_score.item())

    """
    normalized_Embeddings
    """

    video_embeddings_normalized = normalize_embeddings(video_embeddings).clone()
    text_embeddings_normalized = normalize_embeddings(text_embeddings).clone()


    similarity_scores_normalized = th.matmul(
        video_embeddings_normalized, text_embeddings_normalized.t()
    )

    mean_score_normalized = th.mean(similarity_scores_normalized).numpy()
    min_score_normalized = th.min(similarity_scores_normalized).numpy()
    max_score_normalized = th.max(similarity_scores_normalized).numpy()
    std_score_normalized = th.std(similarity_scores_normalized).numpy()
    # print("Summary of normalized embeddings")
    # print("Mean similarity score after norm:", mean_score_normalized.item())
    # print("Min similarity score after norm:", min_score_normalized.item())
    # print("Max similarity score after norm:", max_score_normalized.item())
    # print("STD of similarity scores after norm:", std_score_normalized.item())
    """
    Standardnize
    """

    video_embeddings_stan = standradize_embeddings(video_embeddings).clone()
    text_embeddings_stan = standradize_embeddings(text_embeddings).clone()
    similarity_scores_stan = th.matmul(
        video_embeddings_stan, text_embeddings_stan.t()
    )

    mean_score_stan = th.mean(similarity_scores_stan).numpy()
    min_score_stan = th.min(similarity_scores_stan).numpy()
    max_score_stan = th.max(similarity_scores_stan).numpy()
    std_score_stan = th.std(similarity_scores_stan).numpy()

    print(
        f"TOP K accuracies of UNnormalized embeddings without PCA"
    )
    accuracies_unnorm = check_pairs(video_embeddings.numpy(), text_embeddings.numpy(), mappings, False)
    data_to_write_unnorm = [
        data_type, sample_size, 'Unnormalized',
        accuracies_unnorm.get("Top 1", ""), accuracies_unnorm.get("Top 3", ""),
        accuracies_unnorm.get("Top 5", ""),accuracies_unnorm.get("Top 10", ""),
        mean_score,min_score,max_score,std_score
    ]
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write_unnorm)
    print(
        "TOP K accuracies of normalized embeddings without PCA"
    )
    accuracies_norm = check_pairs(
        video_embeddings_normalized.numpy(),
        text_embeddings_normalized.numpy(),
        mappings,
        False,
    )
    data_to_write_norm = [
        data_type, sample_size, 'Normalized',
        accuracies_norm.get("Top 1", ""), accuracies_norm.get("Top 3", ""),
        accuracies_norm.get("Top 5", ""), accuracies_norm.get("Top 10", ""),
        mean_score_normalized, min_score_normalized, max_score_normalized, std_score_normalized
    ]
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write_norm)

    print(
        "TOP K accuracies of standardized embeddings without PCA"
    )
    accuracies_stan = check_pairs(
        video_embeddings_stan.numpy(),
        text_embeddings_stan.numpy(),
        mappings,
        False,
    )
    data_to_write_stan = [
        data_type, sample_size, 'Standardize',
        accuracies_stan.get("Top 1", ""), accuracies_stan.get("Top 3", ""),
        accuracies_stan.get("Top 5", ""), accuracies_stan.get("Top 10", ""),
        mean_score_stan, min_score_stan, max_score_stan, std_score_stan
    ]
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write_stan)
    """
    PCA PLOT
    """
    # combined_embeddings = np.vstack((video_embeddings, text_embeddings))
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(combined_embeddings)
    #
    # combined_embeddings_norm = np.vstack((video_embeddings_normalized, text_embeddings_normalized))
    # pca_norm = PCA(n_components=2)
    # reduced_embeddings_norm = pca_norm.fit_transform(combined_embeddings_norm)
    #
    # combined_embeddings_stan = np.vstack((video_embeddings_stan, text_embeddings_stan))
    # pca_stan = PCA(n_components=2)
    # reduced_embeddings_stan = pca_stan.fit_transform(combined_embeddings_stan)
    #
    # reduced_video_embeddings = th.from_numpy(reduced_embeddings[: len(video_embeddings)]).float()
    # reduced_text_embeddings = th.from_numpy(reduced_embeddings[len(text_embeddings):]).float()
    #
    # reduced_video_embeddings_norm = th.from_numpy(reduced_embeddings_norm[: len(video_embeddings_normalized)]).float()
    # reduced_text_embeddings_norm = th.from_numpy(reduced_embeddings_norm[len(text_embeddings_normalized):]).float()
    #
    # reduced_video_embeddings_stan = th.from_numpy(reduced_embeddings_stan[: len(video_embeddings_stan)]).float()
    # reduced_text_embeddings_stan = th.from_numpy(reduced_embeddings_stan[len(text_embeddings_stan):]).float()

    # similarity_scores_PCA = th.matmul(
    #     reduced_text_embeddings, reduced_video_embeddings.t()
    # )
    #
    # similarity_scores_PCA_norm = th.matmul(
    #     reduced_text_embeddings_norm, reduced_video_embeddings_norm.t()
    # )

    print(
        f"TOP K accuracies of normalized embeddings with 2D PCA"
    )
    #check_pairs(reduced_video_embeddings_norm.numpy(), reduced_text_embeddings_norm.numpy(), mappings, False)
    plot_embeddings(video_embeddings_normalized,
                    text_embeddings_normalized, mappings, 'plots/taskA/OpenX/norm', f'{data_type}_norm', False)
    print(
        "TOP K accuracies of UNnormalized embeddings with 2D PCA"
    )
    # check_pairs(
    #     reduced_video_embeddings.numpy(),
    #     reduced_text_embeddings.numpy(),
    #     mappings,
    #     False,
    # )
    plot_embeddings(video_embeddings,
                    text_embeddings, mappings, 'plots/taskA//OpenX/unnorm', f'{data_type}_Unnorm', False)
    plot_embeddings(video_embeddings_stan,
                    text_embeddings_stan, mappings, 'plots/taskA//OpenX/stan', f'{data_type}_stan', False)

