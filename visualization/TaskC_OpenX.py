import PIL
from s3dg import S3D
import torch as th
import json
import csv
# from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2
from pca import plot_embeddings_3d, plot_embeddings, check_pairs, plot_distribution_histograms
from mlp import mlp_eval, normalize_embeddings
from Transformation_Matrix import  eval_M2, eval_M
from torch.utils.data import Dataset, DataLoader
from Dataload import VideoTextDataset, Embedding, list_webm_files, filter_top_embeddings, OpenXDataset, SthDataset
import joblib
import argparse
from sklearn.model_selection import train_test_split

START_SAMPLE_SIZE = 50
VAL_SAMPLE_SIZE = 150

if __name__ == "__main__":
    csv_file_path = "SubspaceAlignment_Result_OpenX.csv"
    if not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0:
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Sample Category", "Filter", "Sample Size for Test", "Variance Threshold", "Sample Size for Training",
                 "Model Name", "Dimensions", "Top 1 Accuracy", "Top 3 Accuracy", "Top 5 Accuracy", "Top 10 Accuracy",
                 "Top 1 Mrr", "Top 3 Mrr", "Top 5 Mrr", "Top 10 Mrr", "Mean Simi Score"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=bool, default=False)
    parser.add_argument("--filter", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="droid")
    args = parser.parse_args()
    filter = False
    if args.filter:
        filter = True
    model_name = "Subspace_Alignment"
    if th.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(th.load("../s3d_howto100m.pth"))
    s3d.eval()
    plot_dir_name = f'{args.dataset}_Train_Test'
    if args.validation:
        plot_dir_name = f'{args.dataset}_Validate_Test'

    embedding_pairs = []
    model_labels = []
    variance_thresholds = [0.9, 0.95, 512]
    sample_sizes = np.array([1]) #* START_SAMPLE_SIZE
    for sample_size in sample_sizes:
        video_text_dataset = OpenXDataset(
            f'/scr/yusenluo/RoboCLIP/OpenX/{args.dataset}', random_samples=False, dataset_name=args.dataset
        )
        train_dataset, val_dataset = train_test_split(video_text_dataset, test_size=0.2, random_state=42)
        sample_size_for_training = len(train_dataset)
        if sample_size >= 1024:
            variance_thresholds.append(512)
        if args.validation:
            sample_size_for_test = len(val_dataset)
            data_loader = DataLoader(
                val_dataset, batch_size=50, shuffle=False, num_workers=5
            )
        else:
            sample_size_for_test = len(train_dataset)
            data_loader = DataLoader(
                train_dataset, batch_size=50, shuffle=False, num_workers=5
            )

        (
            train_video_embeddings,
            train_text_embeddings,
            train_embeddings_dataset,
            train_mappings,
        ) = Embedding(s3d, data_loader)
        '''''
        test
        '''''
        train_video_embeddings = 2 * train_text_embeddings - 0.5

        print(f"{plot_dir_name} RESULTS with {sample_size} samples")
        if filter:
            filtered_video_embeddings, filtered_text_embeddings = filter_top_embeddings(train_video_embeddings.clone(),
                                                                          train_text_embeddings,
                                                                          0.5)
            train_video_embeddings_normalized = normalize_embeddings(
                filtered_video_embeddings
            ).clone()
            train_text_embeddings_normalized = normalize_embeddings(
                filtered_text_embeddings
            ).clone()
        else:
            # train_video_embeddings_normalized = normalize_embeddings(
            #     train_video_embeddings
            # ).clone()
            # train_text_embeddings_normalized = normalize_embeddings(
            #     train_text_embeddings
            # ).clone()
            train_video_embeddings_normalized = train_video_embeddings.clone()
            train_text_embeddings_normalized = train_text_embeddings.clone()
        embedding_pairs.append((train_video_embeddings_normalized, train_text_embeddings_normalized))
        model_labels.append('Original Normalized Embeddings space')
        """
        Original Embeddings space
        """

        print(
            f"Normalized result in the {train_video_embeddings_normalized.shape[1]}D Original Embeddings space"
        )

        accuracies_original, mrr_original = check_pairs(
            train_video_embeddings_normalized.numpy(),
            train_text_embeddings_normalized.numpy(),
            train_mappings,
            False,
        )
        simi_score = th.mean(th.diag(th.matmul(train_video_embeddings_normalized, train_text_embeddings_normalized.t())))
        data_to_write_original = [
            plot_dir_name, filter, sample_size_for_test, 'No PCA',
            sample_size_for_training, 'Original', train_video_embeddings_normalized.shape[1],
            accuracies_original.get("Top 1", ""), accuracies_original.get("Top 3", ""), accuracies_original.get("Top 5", ""),
            accuracies_original.get("Top 10", ""), mrr_original.get("Top 1", ""), mrr_original.get("Top 3", ""), mrr_original.get("Top 5", ""),
                mrr_original.get("Top 10", ""), simi_score.item()
        ]
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_to_write_original)
        if filter:
            plot_original = f"plots/taskC/original/filter/{plot_dir_name}"
        else:
            plot_original = f"plots/taskC/original/{plot_dir_name}"
        #full_pca_path = f"saved_model/Full_PCA/pca_model_full_original_{sample_size}.pkl"
        pca = plot_embeddings(
            train_video_embeddings_normalized.numpy(),
            train_text_embeddings_normalized.numpy(),
            train_mappings,
            plot_original,
            f"pca_plot_original_{sample_size_for_training}.png",
            False,
        )
        # if not filter:
        #     joblib.dump(pca, full_pca_path)

        for variance_threshold in variance_thresholds:
            """
            PCA
            """
            if filter:
                pca_video_model_path = (
                    f"saved_model/M/OpenX/droid/alignXtoX/filter/pca_model_video_{variance_threshold}_{sample_size_for_training}.pkl"
                )
                pca_text_model_path = (
                    f"saved_model/M/OpenX/test/alignXtoX/filter/pca_model_text_{variance_threshold}_{sample_size_for_training}.pkl"
                )
            else:
                pca_video_model_path = (
                    f"saved_model/M/OpenX/test/alignXtoX/pca_model_video_{variance_threshold}_{sample_size_for_training}.pkl"
                )
                pca_text_model_path = (
                    f"saved_model/M/OpenX/test/alignXtoX/pca_model_text_{variance_threshold}_{sample_size_for_training}.pkl"
                )

            video_pca = joblib.load(pca_video_model_path)
            text_pca = joblib.load(pca_text_model_path)

            print(
                f"Results with variance_threshold {variance_threshold} and {sample_size}:"
            )
            """
            TOP K accuracies after PCA AFTER M
            """

            video_embeddings_pca = video_pca.transform(
                train_video_embeddings_normalized.clone()
            )
            text_embeddings_pca = text_pca.transform(
                train_text_embeddings_normalized.clone()
            )

            #text_embeddings_pca = np.dot((train_text_embeddings_normalized.clone()).numpy(), (text_pca.components_).T)

            video_embeddings_tensor = (
                th.from_numpy(video_embeddings_pca).float().to(device)
            )
            text_embeddings_tensor = (
                th.from_numpy(text_embeddings_pca).float().to(device)
            )
            if filter:
                M_model_path = f"saved_model/M/OpenX/{args.dataset}/alignXtoX/filter/M_model_{variance_threshold}_{sample_size_for_training}.pth"
                plot_path = f"plots/taskC/M/OpenX/{args.dataset}/alignXtoX/filter/{plot_dir_name}"
            else:
                M_model_path = f"saved_model/M/OpenX/test/alignXtoX/M_model_{variance_threshold}_{sample_size_for_training}.pth"
                plot_path = f"plots/taskC/M/OpenX/test/alignXtoX/{plot_dir_name}"
            adjusted_video_embeddings = eval_M(
                video_embeddings_tensor, M_model_path
            )
            embedding_pairs.append((adjusted_video_embeddings.cpu(), th.from_numpy(text_embeddings_pca)))
            model_labels.append(f'Subspace_alignment_{variance_threshold}_{sample_size_for_training}')
            dimensions = adjusted_video_embeddings.shape[1]
            print(
                f"Result after MLP_{variance_threshold}_{sample_size_for_training} in {dimensions}D space"
            )
            accuracies_Model, mrr_Model = check_pairs(
                adjusted_video_embeddings.cpu().numpy(),
                text_embeddings_pca,
                train_mappings,
                False,
            )
            #full_pca_path = f"saved_model/Full_PCA/pca_model_full_{variance_threshold}_{sample_size}.pkl"
            pca = plot_embeddings(
                adjusted_video_embeddings.cpu().numpy(),
                text_embeddings_pca,
                train_mappings,
                plot_path,
                f"pca_plot_M_{variance_threshold}_{sample_size_for_training}.png",
                False,
            )
            # if not filter:
            #     joblib.dump(pca, full_pca_path)
            simi_score_model = th.mean(
                th.diag(th.matmul(adjusted_video_embeddings.cpu().float(), th.from_numpy(text_embeddings_pca).float().t())))
            data_to_write_model= [
                plot_dir_name, filter, sample_size_for_test, variance_threshold,
                sample_size_for_training, model_name, dimensions,
                accuracies_Model.get("Top 1", ""), accuracies_Model.get("Top 3", ""), accuracies_Model.get("Top 5", ""),
                accuracies_Model.get("Top 10", ""), mrr_Model.get("Top 1", ""), mrr_Model.get("Top 3", ""), mrr_Model.get("Top 5", ""),
                mrr_Model.get("Top 10", ""), simi_score_model.item()
            ]

            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_write_model)
    plot_distribution_histograms(*embedding_pairs, labels=model_labels, dataset_name=plot_dir_name)
