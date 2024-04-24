import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding, filter_top_embeddings, SthDataset, OpenXDataset
from sklearn.model_selection import train_test_split
import torch as th
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D
from sklearn.decomposition import PCA
import joblib
import torch.nn.functional as F
import numpy as np
from mlp import normalize_embeddings
import argparse


def reduce_dimension(
    embeddings, variance_threshold, train_size, embed_type, dimension=None, filter = False
):
    if dimension:
        pca = PCA(n_components=dimension)
    else:
        pca = PCA(n_components=variance_threshold)
    reduced_embeddings = pca.fit_transform(embeddings)
    if filter:
        model_filename = (
            f"saved_model/M/OpenX/droid/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"saved_model/M/OpenX/test/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return pca.components_ , torch.from_numpy(reduced_embeddings).float()


def reduce_dimension_trained(
    embeddings, variance_threshold, train_size, embed_type, filter = False
):
    if filter:
        model_filename = (
            f"saved_model/M/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"saved_model/M/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    pca = joblib.load(model_filename)
    reduced_embeddings = pca.transform(embeddings)

    print(f"Using PCA_{embed_type} from {model_filename}")
    return pca.components_ , torch.from_numpy(reduced_embeddings).float()


def compute_M(video_embeddings, text_embeddings, variance_threshold, train_size, filter):
    # video_embeddings = normalize_embeddings(video_embeddings)
    # text_embeddings = normalize_embeddings(text_embeddings)
    X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', filter=filter) # 35, 512
    X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold,
                                          train_size, 'video', dimension= X_T.shape[0], filter=filter)# 35，512
    M = np.dot(X_S, X_T.T) # 35 35
    M_tensor = torch.from_numpy(M).float()
    if filter:
        save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/filter"
    else:
        save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/test"
    M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
    torch.save(M_tensor, M_model_path)
    print(f'M model saved to {M_model_path}')

# def compute_M_1(video_embeddings, text_embeddings, variance_threshold, train_size):
#     video_embeddings = normalize_embeddings(video_embeddings)
#     text_embeddings = normalize_embeddings(text_embeddings)
#     X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', 512) # 35, 512
#     X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video', X_T.shape[0])# 35，512
#     M = np.dot(X_S, X_T.T) # 35 35
#     A = np.dot(X_S.T, M) # 512 35
#     A_tensor = torch.from_numpy(A).float()
#
#     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M_1/filter"
#     M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
#     torch.save(A_tensor, M_model_path)
#     print(f'M model saved to {M_model_path}')
#
# def compute_M_2(video_embeddings, text_embeddings, variance_threshold, train_size):
#     video_embeddings = normalize_embeddings(video_embeddings)
#     text_embeddings = normalize_embeddings(text_embeddings)
#     X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', 512) # 35, 512
#     X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video', X_T.shape[0])# 35，512
#     M = np.dot(X_S.T, X_T) # 512 512
#     A = np.dot(X_S, M) # 35 512
#     A = A.T #512 35
#     A_tensor = torch.from_numpy(A).float()
#
#     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M_2/filter"
#     M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
#     torch.save(A_tensor, M_model_path)
#     print(f'M model saved to {M_model_path}')
#
# def compute_M_3(video_embeddings, text_embeddings, variance_threshold, train_size):
#     video_embeddings = normalize_embeddings(video_embeddings)
#     text_embeddings = normalize_embeddings(text_embeddings)
#     X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', 512) # 35, 512
#     X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video', X_T.shape[0])# 35，512
#     M = np.dot(X_S, X_S.T) # 35 35
#     A = np.dot(M, X_T) # 35 512
#     A = A.T #512 35
#     A_tensor = torch.from_numpy(A).float()
#     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M_3/filter"
#     M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
#     torch.save(A_tensor, M_model_path)
#     print(f'M model saved to {M_model_path}')


def eval_M(video_embeddings_pca, M_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    #M_model.load_state_dict(torch.load(M_path))
    Matrix = torch.load(M_path).to(device)
    with torch.no_grad():
        adjust_video_embeddings = torch.matmul(video_embeddings_pca, Matrix)
    return adjust_video_embeddings

def eval_M2(video_embeddings_pca, M_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    Matrix = torch.load(M_path).to(device)
    with torch.no_grad():
        adjust_video_embeddings = torch.matmul(video_embeddings_pca, Matrix)
    return adjust_video_embeddings


def similarity_score(adjust_video_embeddings, text_embeddings_pca):

    sim_scores = F.cosine_similarity(adjust_video_embeddings, text_embeddings_pca, dim=1)
    return sim_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=bool, default=False)
    args = parser.parse_args()
    filter = False
    if args.filter:
        filter = True
    variance_thresholds = [0.9, 0.95, 512]
    sample_sizes = [1] #[1, 2, 4, 8, 16, 21]
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths = list_webm_files(
        "../20bn-something-something-v2/train"
    )  # '../20bn-something-something-v2'
   # print(len(video_paths))
    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(torch.load("../s3d_howto100m.pth"))
    s3d.eval()

    for size_multiplier in sample_sizes:
        current_sample_size = 50 * size_multiplier
        if current_sample_size == 1050:
            variance_thresholds.append(512)
        # video_text_dataset = VideoTextDataset(
        #     video_paths, num_samples=current_sample_size, random_samples=False
        # )
        # data_loader = DataLoader(
        #     video_text_dataset, batch_size=50, shuffle=False, num_workers=10
        # )
        # video_text_dataset = SthDataset(
        #         "../20bn-something-something-v2/train", random_samples=False
        #     )
        video_text_dataset = OpenXDataset(
            '/scr/yusenluo/RoboCLIP/OpenX/droid', random_samples=False, dataset_name='droid'
        )
        train_dataset, val_dataset = train_test_split(video_text_dataset, test_size=0.2, random_state=42)
        current_sample_size = len(train_dataset)
        data_loader = DataLoader(
            train_dataset, batch_size=50, shuffle=False, num_workers=5
        )
        video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(
            s3d, data_loader
        )
        '''
        test
        '''
        video_embeddings = 2 * text_embeddings - 0.5
        if filter:
            video_embeddings, text_embeddings = filter_top_embeddings(video_embeddings, text_embeddings, 0.5)

        for variance_threshold in variance_thresholds:
            print(
                f"Training with variance threshold {variance_threshold} and sample size {current_sample_size}."
            )
            compute_M(video_embeddings, text_embeddings, variance_threshold, current_sample_size, filter)
