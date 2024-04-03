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

    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(th.load("../s3d_howto100m.pth"))
    s3d.eval()

    # validate_dataset = VideoTextDataset(
    #     validation_video_paths,
    #     num_samples= VAL_SAMPLE_SIZE,
    #     random_samples=False,
    #     dataset_type="validation",
    # )
    # validate_data_loader = DataLoader(
    #     validate_dataset, batch_size=25, shuffle=True, num_workers=2
    # )
    # (
    #     train_video_embeddings,
    #     train_text_embeddings,
    #     train_embeddings_dataset,
    #     train_mappings,
    # ) = Embedding(s3d, validate_data_loader)


    variance_thresholds = [0.9, 0.95]
    sample_sizes = np.array([1, 2, 4, 8, 16]) * START_SAMPLE_SIZE
    #check_points = [1000, 2000]
    for sample_size in sample_sizes:
        training_dataset = VideoTextDataset(
            training_video_paths,
            num_samples=sample_size,
            random_samples=False,
            dataset_type="train",
        )
        train_data_loader = DataLoader(
            training_dataset, batch_size=25, shuffle=True, num_workers=2
        )
        (
            train_video_embeddings,
            train_text_embeddings,
            train_embeddings_dataset,
            train_mappings,
        ) = Embedding(s3d, train_data_loader)
        print(f"Training RESULTS with {sample_size} samples")
        """
        No PCA BEFORE MLP
        """
        train_video_embeddings_normalized = normalize_embeddings(
            train_video_embeddings
        ).clone()
        train_text_embeddings_normalized = normalize_embeddings(
            train_text_embeddings
        ).clone()

        print(
            f"Normalized result BEFORE MLP without any PCA in the {train_video_embeddings_normalized.shape[1]}D space"
        )
        check_pairs(
            train_video_embeddings_normalized.numpy(),
            train_text_embeddings_normalized.numpy(),
            train_mappings,
            False
        )
        plot_embeddings(train_video_embeddings_normalized.numpy(),
            train_text_embeddings_normalized.numpy(),
            train_mappings,
            "plots/taskB/Train/nomlp",
            f"pca_plot_PCA_{1}_{sample_size}.png",
            False)
        """
        No PCA AFTER MLP
        """
        mlp_model_path = (
            f"saved_model/final_model_{1}_{sample_size}.pth"
        )
        adjusted_video_embeddings = mlp_eval(
            train_video_embeddings_normalized.to(device), train_text_embeddings_normalized.to(device), mlp_model_path
        )
        print(
            f"Normalized result AFTER MLP without any PCA in the {train_video_embeddings_normalized.shape[1]}D space"
        )
        check_pairs(
            adjusted_video_embeddings.cpu().numpy(),
            train_text_embeddings_normalized.cpu().numpy(),
            train_mappings,
            False
        )
        plot_embeddings(adjusted_video_embeddings.cpu().numpy(),
                        train_text_embeddings_normalized.numpy(),
                        train_mappings,
                        "plots/taskB/Train/mlp",
                        f"pca_plot_mlp2D_{1}_{sample_size}.png",
                        False)

        for variance_threshold in variance_thresholds:
            """
            PCA
            """
            pca_video_model_path = (
                f"saved_model/pca_model_video_{variance_threshold}_{sample_size}.pkl"
            )
            video_pca = joblib.load(pca_video_model_path)
            pca_text_model_path = (
                f"saved_model/pca_model_text_{variance_threshold}_{sample_size}.pkl"
            )
            text_pca = joblib.load(pca_text_model_path)
            train_video_embeddings_text_pca = text_pca.transform(
                train_video_embeddings_normalized.clone()
            )
            train_text_embeddings_text_pca = text_pca.transform(
                train_text_embeddings_normalized.clone()
            )
            """
            TOP K accuracies after PCA NO MLP
            """
            print(
                f"Results with variance_threshold {variance_threshold} and {sample_size}:"
            )
            print(
                f"Training result BEFORE MLP with PCA_Text_{variance_threshold} and {sample_size} in {text_pca.n_components_}D space:"
            )
            check_pairs(
                train_video_embeddings_text_pca, train_text_embeddings_text_pca, train_mappings, False
            )
            plot_embeddings(
                train_video_embeddings_text_pca,
                train_text_embeddings_text_pca,
                train_mappings,
                "plots/taskB/Train/nomlp",
                f"pca_plot_PCA_{variance_threshold}_{sample_size}.png",
                False,
            )
            """
            TOP K accuracies after PCA AFTER MLP
            """


            video_embeddings_pca = video_pca.transform(
                train_video_embeddings_normalized.clone()
            )
            text_embeddings_pca = text_pca.transform(train_text_embeddings_normalized.clone())

            video_embeddings_tensor = (
                th.from_numpy(video_embeddings_pca).float().to(device)
            )
            text_embeddings_tensor = (
                th.from_numpy(text_embeddings_pca).float().to(device)
            )

            mlp_model_path = (
                f"saved_model/final_model_{variance_threshold}_{sample_size}.pth"
            )
            adjusted_video_embeddings = mlp_eval(
                video_embeddings_tensor, text_embeddings_tensor, mlp_model_path
            )

            print(
                f"Result after MLP_{variance_threshold}_{sample_size} in {adjusted_video_embeddings.shape[1]}D space"
            )
            check_pairs(
                adjusted_video_embeddings.cpu().numpy(),
                text_embeddings_pca,
                train_mappings,
                False,
            )
            plot_embeddings(
                adjusted_video_embeddings.cpu().numpy(),
                text_embeddings_pca,
                train_mappings,
                "plots/taskB/Train/mlp",
                f"pca_plot_mlp2D_{variance_threshold}_{sample_size}.png",
                False,
            )
            # plot_embeddings_3d(
            #    adjusted_video_embeddings.cpu().numpy(),
            #    text_embeddings,
            #    mappings,
            #    "plots/PCA_mlp",
            #    f"pca_plot_mlp3D_{variance_threshold}_{sample_size}.png",
            #    False,
            # )
