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
from mlp import mlp_eval, normalize_embeddings
from Transformation_Matrix import  eval_M2, eval_M
from torch.utils.data import Dataset, DataLoader
from Dataload import VideoTextDataset, Embedding, list_webm_files, filter_top_embeddings
import joblib
import argparse

START_SAMPLE_SIZE = 50
VAL_SAMPLE_SIZE = 150

if __name__ == "__main__":
    csv_file_path = "WeightVector_Result.csv"
    if not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0:
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Sample Category", "Filter", "Sample Size for Test", "Variance Threshold", "Sample Size for Training",
                 "Model Name", "Dimensions", "Top 1 Accuracy", "Top 3 Accuracy", "Top 5 Accuracy", "Top 10 Accuracy",
                 "Top 1 Mrr", "Top 3 Mrr", "Top 5 Mrr", "Top 10 Mrr", "Mean Simi Score", "Cos Simi"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=bool, default=False)
    args = parser.parse_args()

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
    plot_dir_name = 'Train'
    if args.validation:
        plot_dir_name = 'Validate'
        sample_size_for_test = VAL_SAMPLE_SIZE
        validate_dataset = VideoTextDataset(
            validation_video_paths,
            num_samples= VAL_SAMPLE_SIZE,
            random_samples=False,
            dataset_type="validation",
        )
        validate_data_loader = DataLoader(
            validate_dataset, batch_size=25, shuffle=True, num_workers=2
        )
        (
            train_video_embeddings,
            train_text_embeddings,
            train_embeddings_dataset,
            train_mappings,
        ) = Embedding(s3d, validate_data_loader)

    variance_thresholds = [0.9, 0.95]
    sample_sizes = np.array([1, 2, 4, 8, 16, 21]) * START_SAMPLE_SIZE
    # check_points = [1000, 2000]
    for sample_size in sample_sizes:
        sample_size_for_training = sample_size
        if not args.validation:
            sample_size_for_test = sample_size
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

        print(f"{plot_dir_name} RESULTS with {sample_size} samples")
        # top_video_embeddings, top_text_embeddings = filter_top_embeddings(train_video_embeddings,
        #                                                                   train_text_embeddings,
        #                                                                   0.5)
        train_video_embeddings_normalized = normalize_embeddings(
            train_video_embeddings
        ).clone()
        train_text_embeddings_normalized = normalize_embeddings(
            train_text_embeddings
        ).clone()
        """
        No PCA BEFORE MLP
        """

        print(
            f"Normalized result BEFORE MLP without any PCA in the {train_video_embeddings_normalized.shape[1]}D space"
        )
        accuracies_original = check_pairs(
            train_video_embeddings_normalized.numpy(),
            train_text_embeddings_normalized.numpy(),
            train_mappings,
            False,
        )
        data_to_write_original = [
            plot_dir_name, sample_size_for_test, 'No PCA',
            sample_size_for_training, 'Original', train_video_embeddings_normalized.shape[1],
            accuracies_original.get("Top 1", ""), accuracies_original.get("Top 3", ""),
            accuracies_original.get("Top 5", ""),
            accuracies_original.get("Top 10", "")
        ]
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_to_write_original)
        # plot_embeddings(
        #     train_video_embeddings_normalized.numpy(),
        #     train_text_embeddings_normalized.numpy(),
        #     train_mappings,
        #     f"plots/taskC/original/filter/{plot_dir_name}",
        #     f"pca_plot_original_{sample_size}.png",
        #     False,
        # )
        """
        No PCA AFTER MLP
        """
        mlp_model_path = f"saved_model/weight_vector/final_model_{1}_{sample_size}.pth"
        adjusted_video_embeddings = mlp_eval(
            train_video_embeddings_normalized.to(device),
            train_text_embeddings_normalized.to(device),
            mlp_model_path,
        )
        print(
            f"Normalized result AFTER MLP without any PCA in the {train_video_embeddings_normalized.shape[1]}D space"
        )
        accuracies_mlp_noPCA = check_pairs(
            adjusted_video_embeddings.cpu().numpy(),
            train_text_embeddings_normalized.cpu().numpy(),
            train_mappings,
            False,
        )
        data_to_write_mlp_noPCA = [
            plot_dir_name, sample_size_for_test, 'No PCA',
            sample_size_for_training, 'Weight Vector only', train_video_embeddings_normalized.shape[1],
            accuracies_mlp_noPCA.get("Top 1", ""), accuracies_mlp_noPCA.get("Top 3", ""),
            accuracies_mlp_noPCA.get("Top 5", ""),
            accuracies_mlp_noPCA.get("Top 10", "")
        ]
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_to_write_mlp_noPCA)
        # plot_embeddings(
        #     adjusted_video_embeddings.cpu().numpy(),
        #     train_text_embeddings_normalized.numpy(),
        #     train_mappings,
        #     f"plots/taskB/{plot_dir_name}/mlp",
        #     f"pca_plot_mlp2D_{1}_{sample_size}.png",
        #     False,
        # )

        for variance_threshold in variance_thresholds:
            """
            PCA
            """
            pca_video_model_path = (
                f"saved_model/weight_vector/pca_model_video_{variance_threshold}_{sample_size}.pkl"
            )
            video_pca = joblib.load(pca_video_model_path)
            pca_text_model_path = (
                f"saved_model/weight_vector/pca_model_text_{variance_threshold}_{sample_size}.pkl"
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
            accuracies_PCA = check_pairs(
                train_video_embeddings_text_pca,
                train_text_embeddings_text_pca,
                train_mappings,
                False,
            )
            data_to_write_PCA = [
                plot_dir_name, sample_size_for_test, variance_threshold,
                sample_size_for_training, 'TEXT PCA only', text_pca.n_components_,
                accuracies_PCA.get("Top 1", ""), accuracies_PCA.get("Top 3", ""),
                accuracies_PCA.get("Top 5", ""),
                accuracies_PCA.get("Top 10", "")
            ]
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_write_PCA)
            # plot_embeddings(
            #     train_video_embeddings_text_pca,
            #     train_text_embeddings_text_pca,
            #     train_mappings,
            #     f"plots/taskB/{plot_dir_name}/nomlp",
            #     f"pca_plot_PCA_{variance_threshold}_{sample_size}.png",
            #     False,
            # )
            """
            TOP K accuracies after PCA AFTER MLP
            """

            video_embeddings_pca = video_pca.transform(
                train_video_embeddings_normalized.clone()
            )
            text_embeddings_pca = text_pca.transform(
                train_text_embeddings_normalized.clone()
            )

            video_embeddings_tensor = (
                th.from_numpy(video_embeddings_pca).float().to(device)
            )
            text_embeddings_tensor = (
                th.from_numpy(text_embeddings_pca).float().to(device)
            )

            mlp_model_path = (
                f"saved_model/weight_vector/final_model_{variance_threshold}_{sample_size}.pth"
            )
            adjusted_video_embeddings = mlp_eval(
                video_embeddings_tensor, text_embeddings_tensor, mlp_model_path
            )
            # M_model_path = (
            #     f"saved_model/M/filter/M_model_{variance_threshold}_{sample_size}.pth"
            # )
            # adjusted_video_embeddings = eval_M(
            #     video_embeddings_tensor, M_model_path
            # )

            print(
                f"Result after MLP_{variance_threshold}_{sample_size} in {adjusted_video_embeddings.shape[1]}D space"
            )
            accuracies_mlp = check_pairs(
                adjusted_video_embeddings.cpu().numpy(),
                text_embeddings_pca,
                train_mappings,
                False,
            )
            data_to_write_mlp = [
                plot_dir_name, sample_size_for_test, variance_threshold,
                sample_size_for_training, 'Weight Vector with PCA', adjusted_video_embeddings.shape[1],
                accuracies_mlp.get("Top 1", ""), accuracies_mlp.get("Top 3", ""),
                accuracies_mlp.get("Top 5", ""),
                accuracies_mlp.get("Top 10", "")
            ]
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_write_mlp)
            # plot_embeddings(
            #     adjusted_video_embeddings.cpu().numpy(),
            #     text_embeddings_pca,
            #     train_mappings,
            #     f"plots/taskB/{plot_dir_name}/mlp",
            #     f"pca_plot_mlp2D_{variance_threshold}_{sample_size}.png",
            #     False,
            # )
            # plot_embeddings_3d(
            #    adjusted_video_embeddings.cpu().numpy(),
            #    text_embeddings,
            #    mappings,
            #    "plots/PCA_mlp",
            #    f"pca_plot_mlp3D_{variance_threshold}_{sample_size}.png",
            #    False,
            # )
