from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import os
from scipy.stats import percentileofscore
# from Transformation_Matrix import similarity_score
import joblib
import seaborn as sns
import torch
import torch.nn.functional as F

# from sklearn.metrics import pairwise_distances

def normalize_embeddings(embeddings):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings


def plot_embeddings(
    video_embeddings,
    text_embeddings,
    mappings,
    directory_name="plots",
    file_name="embeddings_plot.png",
    small_scale=True,
    stats_info=None,
):
    """
    Plots 2D PCA visualizations of video and text embeddings, saving the plot to a file.

    Parameters:
    - video_embeddings (Tensor): The embeddings for video data, to be visualized in blue.
    - text_embeddings (Tensor): The embeddings for text data, to be visualized in red.
    - directory_name (str): The name of the directory where the plot image will be saved.
    - file_name (str): The name of the file to save the plot image as.

    Returns:
    - None. The plot is saved to the specified file.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    if isinstance(video_embeddings, th.Tensor):
        if video_embeddings.device.type == "cuda":
            video_embeddings = video_embeddings.cpu()
        video_embeddings = video_embeddings.detach().numpy()
    if isinstance(text_embeddings, th.Tensor):
        if text_embeddings.device.type == "cuda":
            text_embeddings = text_embeddings.cpu()
        text_embeddings = text_embeddings.detach().numpy()

    num_samples = len(video_embeddings)

    combined_embeddings = np.vstack((video_embeddings, text_embeddings))
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    reduced_video_embeddings = reduced_embeddings[: len(video_embeddings)]
    reduced_text_embeddings = reduced_embeddings[len(video_embeddings) :]

    similarities = video_embeddings @ text_embeddings.T
    self_similarities = np.diag(similarities)
    percentiles = [percentileofscore(self_similarities, sim, 'rank') for sim in self_similarities]

    # pca_video = PCA(n_components=2)
    # reduced_video_embeddings = pca_video.fit_transform(video_embeddings)
    #
    # pca_text = PCA(n_components=2)
    # reduced_text_embeddings = pca_text.fit_transform(text_embeddings)

    plt.figure(figsize=(10, 6))

    # if small_scale:
    #     cmap = plt.get_cmap("Set3")
    #     colors = [cmap(i) for i in range(num_samples)]
    # else:
    #     cmap = plt.get_cmap("inferno")
    #     colors = [cmap(i) for i in np.linspace(0, 1, num_samples)]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(p / 100) for p in percentiles]

    for i in range(num_samples):
        sc = plt.scatter(
            reduced_video_embeddings[i, 0],
            reduced_video_embeddings[i, 1],
            color=colors[i],
            marker="+",
            s=100,
            label=f"Video_Embeddings" if i == 0 else "",
            alpha=1,
        )
        plt.scatter(
            reduced_text_embeddings[i, 0],
            reduced_text_embeddings[i, 1],
            color=colors[i],
            marker="o",
            s=100,
            label=f"Text_Embeddings" if i == 0 else "",
            alpha=1,
        )

    if stats_info is not None:
        plt.text(
            0.05,
            0.95,
            stats_info,
            verticalalignment="top",
            horizontalalignment="left",
            transform=plt.gca().transAxes,
            color="black",
            fontsize=10,
        )
    plt.colorbar(sc, label='Similarity Percentile')
    plt.title("PCA of Video and Text Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.legend()

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    #print("Results in 2D space")
    # check_pairs(
    #     reduced_video_embeddings, reduced_text_embeddings, mappings, small_scale
    # )
    plt.close()
    plt.figure(figsize=(12, 8))

    norms1 = np.linalg.norm(video_embeddings, axis=1)
    norms2 = np.linalg.norm(text_embeddings, axis=1)
    # print(norms1)
    # print(norms2)
    if np.any((norms1 < 0) | (norms2 < 0)):
        print("There are negative values in the norms")
    else:
        print("All norms are non-negative.")
    plt.boxplot([norms1, norms2], labels = ['Video', 'Text'])
    plt.title('Distribution of Each Feature in Embeddings')
    plt.ylabel('Norm Value')
    save_path = os.path.join(directory_name, 'BoxPlot.png')
    plt.savefig(save_path)
    return pca


def normalize_embeddings_torch(embeddings):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings


def reduce_dimension(
        embeddings, variance_threshold, train_size, embed_type, seed, strong, num, dimension=None, filter=False
):
    if variance_threshold == 0:
        return None, embeddings.float()

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
            f"saved_model/M/OpenX/droid/pca_model/pca_model_{embed_type}_{variance_threshold}_{train_size}_Seed{seed}_{strong}_{num}.pkl"
        )
    # joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return pca, torch.from_numpy(reduced_embeddings).float()



def normalize_and_pca(sampled_video_embeddings, sampled_text_embeddings, validate_video_embeddings_normalized, validate_text_embeddings_normalized, variance_threshold, 
                      current_sample_size, seed, device, strong, pca_sample_size):
    train_video_embeddings_normalized = normalize_embeddings(
        sampled_video_embeddings
    ).clone().cpu()
    train_text_embeddings_normalized = normalize_embeddings(
        sampled_text_embeddings
    ).clone().cpu()
    pca_text, reduced_train_text = reduce_dimension(train_text_embeddings_normalized, variance_threshold,
                                                    current_sample_size,
                                                    'text', seed=seed, strong=strong, num=pca_sample_size,
                                                    filter=False)
    pca_video, reduced_train_video = reduce_dimension(train_video_embeddings_normalized, variance_threshold,
                                                      current_sample_size, 'video', filter=False,
                                                      dimension=reduced_train_text.shape[1],
                                                      seed=seed, strong=strong, num=pca_sample_size,)  # 35，512
    if pca_text != None:
        reduced_validate_video = torch.from_numpy(
            pca_video.transform(validate_video_embeddings_normalized.cpu())).float().to(device)
        reduced_validate_text = torch.from_numpy(
            pca_text.transform(validate_text_embeddings_normalized.cpu())).float().to(device)
    else:
        reduced_validate_video = (validate_video_embeddings_normalized).float().to(device)
        reduced_validate_text = (validate_text_embeddings_normalized).float().to(device)
    reduced_train_text = reduced_train_text.to(device)
    reduced_train_video = reduced_train_video.to(device)
    return reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text



def reduce_with_pca_aug(augmented_video_embeddings, augmented_text_embeddings, validate_video_embeddings_normalized,
                                      validate_text_embeddings_normalized, pca_video, pca_text, device):
    train_video_embeddings_normalized = normalize_embeddings_torch(
        augmented_video_embeddings
    ).clone().cpu()
    train_text_embeddings_normalized = normalize_embeddings_torch(
        augmented_text_embeddings
    ).clone().cpu()
    reduced_validate_video = torch.from_numpy(
        pca_video.transform(validate_video_embeddings_normalized.cpu())).float().to(device)
    reduced_validate_text = torch.from_numpy(
        pca_text.transform(validate_text_embeddings_normalized.cpu())).float().to(device)
    reduced_train_video = torch.from_numpy(
        pca_video.transform(train_video_embeddings_normalized)).float().to(device)
    reduced_train_text = torch.from_numpy(
        pca_text.transform(train_text_embeddings_normalized)).float().to(device)
    return reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text

def reduce_with_pca(validate_video_embeddings_normalized, validate_text_embeddings_normalized, pca_video, pca_text, device):

    reduced_validate_video = torch.from_numpy(
        pca_video.transform(validate_video_embeddings_normalized.cpu())).float().to(device)
    reduced_validate_text = torch.from_numpy(
        pca_text.transform(validate_text_embeddings_normalized.cpu())).float().to(device)

    return reduced_validate_video, reduced_validate_text