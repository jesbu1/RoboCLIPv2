from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import os


def check_pairs(reduced_video_embeddings, reduced_text_embeddings, mappings, small_scale = True):
    """
        Checks the pairing accuracy between video and text embeddings by finding
        the nearest text embedding for each video embedding and see whether it is the corresponding text of the video.

        Prints the accuracy of correct pairings and the indices of video embeddings that are incorrectly paired.
        """
    distances = np.linalg.norm(reduced_video_embeddings[:, np.newaxis, :] - reduced_text_embeddings[np.newaxis, :, :],
                               axis=2)
    sorted_text_indices = np.argsort(distances, axis=1)

    video_id_to_text_label = mappings["video_id_to_text_label"]
    index_to_video_id = mappings["index_to_video_id"]
    index_to_text_label = mappings["index_to_text_label"]

    if small_scale:
        ground_truth_indices = np.arange(len(reduced_video_embeddings))
        for n in [1, 3, 5]:
            correct_pairs = np.array(
                [ground_truth in sorted_text_indices[i, :n] for i, ground_truth in enumerate(ground_truth_indices)])
            accuracy = np.mean(correct_pairs)
            print(f"Top {n} accuracy: {accuracy * 100:.2f}%")
        top_1_indices = sorted_text_indices[:, 0]
        correct_top_1_pairs = top_1_indices == ground_truth_indices
        incorrect_top_1_indices = np.where(~correct_top_1_pairs)[0]

        incorrect_pair_video_id = [mappings["index_to_video_id"][i] for i in incorrect_top_1_indices]
        print(f"IDs of incorrectly paired video embeddings (Top 1): {incorrect_pair_video_id}")

        for i, indices in enumerate(sorted_text_indices):
            video_id = index_to_video_id[i]
            original_text_label = video_id_to_text_label[video_id]
            sorted_text_labels = [index_to_text_label[j] for j in indices]
            print(f"Video {video_id}:")
            print(f"Ground truth text label: {original_text_label}")
            print(f"Sorted Matching text labels: {sorted_text_labels}")
    else:
        ground_truth_indices = np.arange(len(reduced_video_embeddings))
        cumulative_accuracy = np.zeros(len(reduced_video_embeddings))
        top_n_values = [1, 3, 5, 10]
        for n in top_n_values:
            for i in range(len(sorted_text_indices)):
                if ground_truth_indices[i] in sorted_text_indices[i, :n]:
                    cumulative_accuracy[i] = 1
            accuracy_n = np.mean(cumulative_accuracy)
            print(f"Accuracy within top {n}: {accuracy_n * 100:.2f}%")

def plot_embeddings(video_embeddings, text_embeddings, mappings, directory_name='plots', file_name='embeddings_plot.png', small_scale = True,stats_info = None):
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

    reduced_video_embeddings = reduced_embeddings[:len(video_embeddings)]
    reduced_text_embeddings = reduced_embeddings[len(video_embeddings):]

    # pca_video = PCA(n_components=2)
    # reduced_video_embeddings = pca_video.fit_transform(video_embeddings)
    #
    # pca_text = PCA(n_components=2)
    # reduced_text_embeddings = pca_text.fit_transform(text_embeddings)

    plt.figure(figsize=(10, 6))

    if small_scale:
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in range(num_samples)]
    else:
        cmap = plt.get_cmap('inferno')
        colors = [cmap(i) for i in np.linspace(0, 1, num_samples)]

    for i in range(num_samples):
        plt.scatter(reduced_video_embeddings[i, 0], reduced_video_embeddings[i, 1],
                   color=colors[i], marker='+', s=100, label = f"Video_Embeddings" if i == 0 else "", alpha = 1)
        plt.scatter(reduced_text_embeddings[i, 0], reduced_text_embeddings[i, 1],
                   color=colors[i], marker='o', s=100, label = f"Text_Embeddings" if i == 0 else "", alpha = 1)

    if stats_info is not None:
        plt.text(0.05, 0.95, stats_info,
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes,
                 color='black', fontsize=10)

    plt.title('PCA of Video and Text Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.legend()

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    print("Results in 2D space")
    check_pairs(reduced_video_embeddings, reduced_text_embeddings, mappings, small_scale)
    plt.close()


def plot_embeddings_3d(video_embeddings, text_embeddings, mappings, directory_name='plots', file_name='3d_embeddings_plot.png', small_scale = True, stats_info = None):
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

    reduced_video_embeddings = reduced_embeddings[:len(video_embeddings)]
    reduced_text_embeddings = reduced_embeddings[len(video_embeddings):]

    # pca_video = PCA(n_components=3)
    # reduced_video_embeddings = pca_video.fit_transform(video_embeddings)
    #
    # pca_text = PCA(n_components=3)
    # reduced_text_embeddings = pca_text.fit_transform(text_embeddings)


    if small_scale:
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in range(num_samples)]
    else:
        cmap = plt.get_cmap('inferno')
        colors = [cmap(i) for i in np.linspace(0, 1, num_samples)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_samples):
        ax.scatter(reduced_video_embeddings[i, 0], reduced_video_embeddings[i, 1], reduced_video_embeddings[i, 2],
                   color=colors[i], marker='+', s=60, label = f"Video_Embeddings" if i == 0 else "", alpha = 1)
        ax.scatter(reduced_text_embeddings[i, 0], reduced_text_embeddings[i, 1], reduced_text_embeddings[i, 2],
                   color=colors[i], marker='o', s=60, label = f"Text_Embeddings" if i == 0 else "", alpha = 1)

    ax.legend()
    ax.set_title('3D PCA of Video and Text Embeddings')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    if stats_info is not None:
        ax.text2D(0.05, 0.95, stats_info,
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes,
                 color='black', fontsize=10)

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    print("Results in 3D space")
    check_pairs(reduced_video_embeddings, reduced_text_embeddings, mappings, small_scale)
    plt.close()