from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch as th
import numpy as np
import os


def check_pairs(reduced_video_embeddings, reduced_text_embeddings):
    """
        Checks the pairing accuracy between video and text embeddings by finding
        the nearest text embedding for each video embedding and see whether it is the corresponding text of the video.

        Prints the accuracy of correct pairings and the indices of video embeddings that are incorrectly paired.
        """
    distances = np.linalg.norm(reduced_video_embeddings[:, np.newaxis, :] - reduced_text_embeddings[np.newaxis, :, :],
                               axis=2)

    nearest_text_indices = np.argmin(distances, axis=1)
    correct_pairs = nearest_text_indices == np.arange(len(reduced_video_embeddings))

    accuracy = np.mean(correct_pairs)
    print(f"Pairing accuracy: {accuracy * 100:.2f}%")

    incorrect_pair_indices = np.where(~correct_pairs)[0]
    print(f"Indices of incorrectly paired video embeddings: {incorrect_pair_indices}")

def plot_embeddings(video_embeddings, text_embeddings, stats_info = None, directory_name='plots', file_name='embeddings_plot.png'):
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

    plt.figure(figsize=(10, 6))

    # cmap = get_cmap('inferno')
    # colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    #
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in range(num_samples)]

    for i in range(num_samples):
        plt.scatter(reduced_video_embeddings[i, 0], reduced_video_embeddings[i, 1],
                   color=colors[i], marker='+', s=100, label = f"Video_Embeddings" if i == 0 else "", alpha = 1)
        plt.scatter(reduced_text_embeddings[i, 0], reduced_text_embeddings[i, 1],
                   color=colors[i], marker='o', s=100, label = f"Text_Embeddings" if i == 0 else "", alpha = 1)

    # num_samples = len(video_embeddings)
    # colors = plt.cm.jet(np.linspace(0, 1, num_samples))
    if stats_info != None:
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
    check_pairs(reduced_video_embeddings, reduced_text_embeddings)
    plt.close()


def plot_embeddings_3d(video_embeddings, text_embeddings, stats_info = None, directory_name='plots', file_name='3d_embeddings_plot.png'):
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

    # cmap = get_cmap('inferno')
    # colors = [cmap(i) for i in np.linspace(0, 1, 12)]

    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in range(num_samples)]

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

    if stats_info != None:
        ax.text2D(0.05, 0.95, stats_info,
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes,
                 color='black', fontsize=10)

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    check_pairs(reduced_video_embeddings, reduced_text_embeddings)
    plt.close()