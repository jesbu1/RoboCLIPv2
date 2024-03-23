from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch as th
import numpy as np
import os

def generate_shaded_colors(base_color_rgb, num_variants):
    """
    Generates a list of shaded colors based on a base RGB color.

    Parameters:
    - base_color_rgb (tuple): A tuple representing the base color in RGB format (e.g., (1, 0, 0) for red).
    - num_variants (int): The number of shaded color variants to generate.

    Returns:
    - list: A list of color variants, from darker to lighter shades of the base color.
    """
    return [(np.array(base_color_rgb) * shade).tolist()
            for shade in np.linspace(0.4, 1, num_variants)]

def plot_embeddings(video_embeddings, text_embeddings, directory_name='plots', file_name='embeddings_plot.png'):
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

    combined_embeddings = np.vstack((video_embeddings, text_embeddings))
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    reduced_video_embeddings = reduced_embeddings[:len(video_embeddings)]
    reduced_text_embeddings = reduced_embeddings[len(video_embeddings):]

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_video_embeddings[:, 0], reduced_video_embeddings[:, 1], c='blue', label='Video Embeddings',
                alpha=0.5)
    plt.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], c='red', label='Text Embeddings',
                alpha=0.5)

    # num_samples = len(video_embeddings)
    # colors = plt.cm.jet(np.linspace(0, 1, num_samples))
    #
    # plt.figure(figsize=(10, 6))
    # for i in range(num_samples):
    #     plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=colors[i], label=f'Sample {i+1} Video', alpha=0.5)
    #     plt.scatter(reduced_embeddings[num_samples + i, 0], reduced_embeddings[num_samples + i, 1], color=colors[i], label=f'Sample {i+1} Text', alpha=0.5)

    plt.title('PCA of Video and Text Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.legend()

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_embeddings_3d(video_embeddings, text_embeddings, directory_name='plots', file_name='3d_embeddings_plot.png'):
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

    # blue_rgb = (0, 0, 1)
    # red_rgb = (1, 0, 0)
    #
    # video_colors = generate_shaded_colors(blue_rgb, num_samples)
    # text_colors = generate_shaded_colors(red_rgb, num_samples)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reduced_video_embeddings[:, 0], reduced_video_embeddings[:, 1], reduced_video_embeddings[:, 2], c='blue',
               label='Video Embeddings')
    ax.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], reduced_text_embeddings[:, 2], c='red',
               label='Text Embeddings')

    # for i, (embed, color) in enumerate(zip(reduced_embeddings[:num_samples], video_colors)):
    #     ax.scatter(embed[0], embed[1], embed[2], color=color, label=f'Video Sample {i + 1}' if i == 0 else "",
    #                alpha=0.5)
    #
    # for i, (embed, color) in enumerate(zip(reduced_embeddings[num_samples:], text_colors)):
    #     ax.scatter(embed[0], embed[1], embed[2], color=color, label=f'Text Sample {i + 1}' if i == 0 else "", alpha=0.5)

    ax.legend()
    ax.set_title('3D PCA of Video and Text Embeddings')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    save_path = os.path.join(directory_name, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()