from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import os
from scipy.stats import percentileofscore
import joblib
import seaborn as sns


def check_pairs(
        reduced_video_embeddings: np.array,
        reduced_text_embeddings: np.array,
        mappings,
        small_scale=False,
        user='text'
):
    if user == 'text':
        similarities = reduced_text_embeddings @ reduced_video_embeddings.T
        k_list = [1, 3, 5, 10]
    else:
        similarities = reduced_video_embeddings @ reduced_text_embeddings.T
        k_list = [15, 45, 75]
    sorted_video_indices = np.argsort(-similarities, axis=1)

    video_id_to_text_label = mappings["video_id_to_text_label"]
    index_to_video_id = mappings["index_to_video_id"]
    index_to_text_label = mappings["index_to_text_label"]
    accuracies = {}
    mrr = {}

    ground_truth_labels = [video_id_to_text_label[index_to_video_id[i]] for i in range(len(reduced_video_embeddings))]

    mrr_k = mrr_score(similarities, ground_truth_labels, index_to_text_label, k_list=k_list)
    print("MRR Scores:", mrr_k)

    if small_scale:
        for n in [1, 3, 5]:
            correct_pairs = np.array(
                [
                    video_id_to_text_label[index_to_video_id[ground_truth]] in [index_to_text_label[idx] for idx in sorted_video_indices[i, :n]]
                    for i, ground_truth in enumerate(range(len(reduced_video_embeddings)))
                ]
            )
            accuracy = np.mean(correct_pairs)
            accuracies[f"Top {n}"] = round(accuracy * 100, 4)
            print(f"Top {n} accuracy: {accuracy * 100:.2f}%")
        top_1_indices = sorted_video_indices[:, 0]
        correct_top_1_pairs = [video_id_to_text_label[index_to_video_id[i]] == index_to_text_label[top_1_idx] for i, top_1_idx in enumerate(top_1_indices)]
        incorrect_top_1_indices = np.where(~np.array(correct_top_1_pairs))[0]

        incorrect_pair_text_id = [
            index_to_text_label[i] for i in incorrect_top_1_indices
        ]
        print(
            f"IDs of incorrectly paired text embeddings (Top 1): {incorrect_pair_text_id}"
        )

        for i, indices in enumerate(sorted_video_indices):
            text_id = index_to_text_label[i]
            original_video_id = video_id_to_text_label[text_id]
            sorted_video_labels = [index_to_video_id[j] for j in indices]
            print(f"Text {text_id}:")
            print(f"Ground truth video id: {original_video_id}")
            print(f"Sorted Matching video ids: {sorted_video_labels}")
    else:
        correct_pairs_info = []

        for n in k_list: #[1, 3, 5, 10]
            correct_pairs = []
            for i, indices in enumerate(sorted_video_indices):
                sorted_video_labels = [index_to_text_label[j] for j in indices[:n]]
                if index_to_text_label[i] in sorted_video_labels:
                    correct_pairs.append(True)
                    correct_pairs_info.append((index_to_text_label[i], ground_truth_labels[i]))
                else:
                    correct_pairs.append(False)
            accuracy = np.mean(correct_pairs)
            accuracies[f"Top {n}"] = round(accuracy * 100, 4)
            print(f"Accuracy within top {n}: {accuracy * 100:.2f}%")

    return accuracies, mrr_k

def mrr_score(similarities, ground_truth_labels, index_to_text_label, k_list=[1, 3, 5, 10]):
    """
    Calculate the Mean Reciprocal Rank (MRR) at multiple values of k for a set of embeddings.

    Parameters:
    - similarities (np.array): A 2D numpy array where each row represents the similarity scores
      between a video embedding and all text embeddings.
    - ground_truth_labels (list): A list where each element is the ground truth text label
      for the corresponding video embedding.
    - index_to_text_label (dict): A dictionary mapping from indices to text labels.
    - k_list (list): A list of integers specifying the k values to calculate MRR for.

    Returns:
    - dict: A dictionary where keys are k values and values are the MRR at each k.
    """
    sorted_indices = np.argsort(-similarities, axis=1)
    mrr_scores = {}

    for k in k_list:
        reciprocal_ranks = []

        for i, sorted_index_list in enumerate(sorted_indices):
            sorted_text_labels = [index_to_text_label[idx] for idx in sorted_index_list[:k]]
            if ground_truth_labels[i] in sorted_text_labels:
                rank = sorted_text_labels.index(ground_truth_labels[i]) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        mrr_scores[f"Top {k}"] = np.mean(reciprocal_ranks)

    return mrr_scores






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
        video_embeddings = video_embeddings.numpy()
    if isinstance(text_embeddings, th.Tensor):
        text_embeddings = text_embeddings.numpy()

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
    norms_v = np.linalg.norm(video_embeddings, axis=1)
    norms_t = np.linalg.norm(text_embeddings, axis=1)
    print(f"Norms of video embeddings after model:{norms_v}")
    print(f"Norms of text embeddings after model:{norms_t}")
    if np.any((norms_v < 0) | (norms_t < 0)):
        print("There are negative values in the norms")
    else:
        print("All norms are non-negative.")
    plt.boxplot([norms_v, norms_t], labels = ['Video', 'Text'])
    plt.title('Distribution of Each Feature in Embeddings')
    plt.ylabel('Norm Value')
    save_path = os.path.join(directory_name, 'BoxPlot.png')
    plt.savefig(save_path)
    return pca


def plot_embeddings_3d(
    video_embeddings,
    text_embeddings,
    mappings,
    directory_name="plots",
    file_name="3d_embeddings_plot.png",
    small_scale=True,
    stats_info=None,
):
    """
    Plots 3D PCA visualizations of video and text embeddings, saving the plot to a file.

    Parameters:
    - video_embeddings (Tensor): The embeddings for video data, to be visualized in blue.
    - text_embeddings (Tensor): The embeddings for text data, to be visualized in red.
    - directory_name (str): The name of the directory where the plot image will be saved.
    - file_name (str): The name of the file to save the plot image as.

    Returns:
    - None. The 3D plot is saved to the specified file.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    if isinstance(video_embeddings, th.Tensor):
        video_embeddings = video_embeddings.detach().numpy()
    if isinstance(text_embeddings, th.Tensor):
        text_embeddings = text_embeddings.detach().numpy()

    num_samples = len(video_embeddings)

    pca = PCA(n_components=3)
    combined_embeddings = np.vstack((video_embeddings, text_embeddings))
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    reduced_video_embeddings = reduced_embeddings[: len(video_embeddings)]
    reduced_text_embeddings = reduced_embeddings[len(video_embeddings) :]

    # pca_video = PCA(n_components=3)
    # reduced_video_embeddings = pca_video.fit_transform(video_embeddings)
    #
    # pca_text = PCA(n_components=3)
    # reduced_text_embeddings = pca_text.fit_transform(text_embeddings)

    if small_scale:
        cmap = plt.get_cmap("Set3")
        colors = [cmap(i) for i in range(num_samples)]
    else:
        cmap = plt.get_cmap("inferno")
        colors = [cmap(i) for i in np.linspace(0, 1, num_samples)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(num_samples):
        ax.scatter(
            reduced_video_embeddings[i, 0],
            reduced_video_embeddings[i, 1],
            reduced_video_embeddings[i, 2],
            color=colors[i],
            marker="+",
            s=60,
            label=f"Video_Embeddings" if i == 0 else "",
            alpha=1,
        )
        ax.scatter(
            reduced_text_embeddings[i, 0],
            reduced_text_embeddings[i, 1],
            reduced_text_embeddings[i, 2],
            color=colors[i],
            marker="o",
            s=60,
            label=f"Text_Embeddings" if i == 0 else "",
            alpha=1,
        )

    ax.legend()
    ax.set_title("3D PCA of Video and Text Embeddings")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    if stats_info is not None:
        ax.text2D(
            0.05,
            0.95,
            stats_info,
            verticalalignment="top",
            horizontalalignment="left",
            transform=plt.gca().transAxes,
            color="black",
            fontsize=10,
        )

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # print("Results in 3D space")
    # check_pairs(
    #     reduced_video_embeddings, reduced_text_embeddings, mappings, small_scale
    # )
    plt.close()





def plot_distribution_histograms(*embeddings_pairs, labels, dataset_name):
    plt.figure(figsize=(12, 8))
    for (video_embeddings, text_embeddings), label in zip(embeddings_pairs, labels):
        video_embeddings = video_embeddings.float()
        text_embeddings = text_embeddings.float()
        similarity = video_embeddings @ text_embeddings.T
        true_match_similarity = np.diag(similarity)
        sns.histplot(true_match_similarity, bins=20, kde=True, label=label, element='step', stat='density')
    plt.title('Distribution of Similarity to True Match Across Different Models')
    plt.xlabel('Similarity to True Match')
    plt.ylabel('Density')
    plt.legend(title='Embedding Types')
    plt.grid(True)
    save_path = f"/scr/yusenluo/RoboCLIP/visualization/plots/taskC/similarity_distribution/{dataset_name}.png"
    plt.savefig(save_path)
    plt.close()


def plot_distribution_histograms_A(*embeddings_pairs, labels, dataset_name):
    plt.figure(figsize=(12, 8))
    for (video_embeddings, text_embeddings), label in zip(embeddings_pairs, labels):
        video_embeddings = video_embeddings.float()
        text_embeddings = text_embeddings.float()
        similarity = text_embeddings @ video_embeddings.T
        true_match_similarity = np.diag(similarity)
        sns.histplot(true_match_similarity, bins=20, kde=True, label=label, element='step', stat='density')
    plt.title('Distribution of Similarity to True Match Across Different Models')
    plt.xlabel('Similarity to True Match')
    plt.ylabel('Density')
    plt.legend(title='Embedding Types')
    plt.grid(True)
    save_path = f"/scr/yusenluo/RoboCLIP/visualization/plots/taskC/similarity_distribution/{dataset_name}.png"
    plt.savefig(save_path)
    plt.close()


def check_pairs_A(
    reduced_video_embeddings: np.array,
    reduced_text_embeddings: np.array,
    mappings,
    similarities,
    small_scale=True,
):
    """
    Checks the pairing accuracy between video and text embeddings by finding
    the nearest text embedding for each video embedding and see whether it is the corresponding text of the video.

    Prints the accuracy of correct pairings and the indices of video embeddings that are incorrectly paired.
    """
    #similarities = reduced_video_embeddings @ reduced_text_embeddings.T
    #similarities = reduced_text_embeddings @ reduced_video_embeddings.T
    #similarities = (similarity_score(reduced_video_embeddings, reduced_text_embeddings)).numpy()
    sorted_text_indices = np.argsort(-similarities, axis=1)

    video_id_to_text_label = mappings["video_id_to_text_label"]
    index_to_video_id = mappings["index_to_video_id"]
    index_to_text_label = mappings["index_to_text_label"]
    accuracies = {}
    mrr = {}
    if small_scale:
        ground_truth_indices = np.arange(len(reduced_video_embeddings))
        mrr_k = mrr_score(similarities, ground_truth_indices, k_list=[1, 3, 5])
        for n in [1, 3, 5]:
            correct_pairs = np.array(
                [
                    ground_truth in sorted_text_indices[i, :n]
                    for i, ground_truth in enumerate(ground_truth_indices)
                ]
            )
            accuracy = np.mean(correct_pairs)
            accuracies[f"Top {n}"] = round(accuracy * 100, 4)
            print(f"Top {n} accuracy: {accuracy * 100:.2f}%")
        top_1_indices = sorted_text_indices[:, 0]
        correct_top_1_pairs = top_1_indices == ground_truth_indices
        incorrect_top_1_indices = np.where(~correct_top_1_pairs)[0]

        incorrect_pair_video_id = [
            mappings["index_to_video_id"][i] for i in incorrect_top_1_indices
        ]
        print(
            f"IDs of incorrectly paired video embeddings (Top 1): {incorrect_pair_video_id}"
        )

        for i, indices in enumerate(sorted_text_indices):
            video_id = index_to_video_id[i]
            original_text_label = video_id_to_text_label[video_id]
            sorted_text_labels = [index_to_text_label[j] for j in indices]
            print(f"Video {video_id}:")
            print(f"Ground truth text label: {original_text_label}")
            print(f"Sorted Matching text labels: {sorted_text_labels}")
    else:
        ground_truth_indices = np.arange(len(reduced_video_embeddings))
        mrr_k = mrr_score(similarities, ground_truth_indices, k_list = [1,3,5,10])
        cumulative_accuracy = np.zeros(len(reduced_video_embeddings))
        top_n_values = [1, 3, 5, 10]
        for n in top_n_values:
            for i in range(len(sorted_text_indices)):
                if ground_truth_indices[i] in sorted_text_indices[i, :n]:
                    cumulative_accuracy[i] = 1
            accuracy_n = np.mean(cumulative_accuracy)
            accuracies[f"Top {n}"] = round(accuracy_n * 100, 4)
            print(f"Accuracy within top {n}: {accuracy_n * 100:.2f}%")
    return accuracies, mrr_k
