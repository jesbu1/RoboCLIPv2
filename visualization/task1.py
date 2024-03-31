import PIL
from s3dg import S3D
import torch as th
import json
#from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2
from pca import plot_embeddings_3d,plot_embeddings
from mlp import mlp_eval
from torch.utils.data import Dataset, DataLoader
from Dataload import VideoTextDataset, Embedding, list_webm_files




def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)

    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if asNumpy is false, a list if PIL images.

    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: ' + str(filename))

    # Load file using PIL
    pilIm = PIL.Image.open(filename)
    pilIm.seek(0)

    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert()  # Make without palette
            a = np.asarray(tmp)
            if len(a.shape) == 0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell() + 1)
    except EOFError:
        pass

    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:
            images.append(PIL.Image.fromarray(im))

    # Done
    return images











if __name__ == '__main__':
    # test_frame = readGif('gifs/dense_original.gif')
    # test_frame = preprocess_human_demo(test_frame)
    # video = th.from_numpy(test_frame)
    # print(test_frame.shape)
    video_paths = list_webm_files('vidz4jesse') #'../20bn-something-something-v2'
    #print(video_paths)
    video_text_dataset = VideoTextDataset(video_paths)
    data_loader = DataLoader(video_text_dataset, batch_size=4, shuffle=True, num_workers=2)

    s3d = S3D('../s3d_dict.npy', 512)
    s3d.load_state_dict(th.load('../s3d_howto100m.pth'))
    s3d.eval()
    video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(s3d, data_loader)
    embeddings_loader = DataLoader(embeddings_dataset, batch_size= 4, shuffle=False)
    #print(video_embeddings.shape, text_embeddings.shape)
    l2_distances = th.norm(text_embeddings - video_embeddings, p=2, dim=1)
    similarity_scores = th.matmul(text_embeddings, video_embeddings.t())
    video_variances = th.var(video_embeddings, dim=0)
    text_variances = th.var(text_embeddings, dim=0)
    mean_video_variance = th.mean(video_variances)
    mean_text_variance = th.mean(text_variances)

    print(f"Mean variance in video embeddings: {mean_video_variance}")
    print(f"Mean variance in text embeddings: {mean_text_variance}")

    mean_distance = th.mean(l2_distances)
    std_distance = th.std(l2_distances)
    min_distance = th.min(l2_distances)
    max_distance = th.max(l2_distances)

    mean_score = th.mean(similarity_scores)
    min_score = th.min(similarity_scores)
    max_score = th.max(similarity_scores)
    std_score = th.std(similarity_scores)

    print("Mean similarity score:", mean_score.item())
    print("Min similarity score:", min_score.item())
    print("Max similarity score:", max_score.item())
    print("STD of similarity scores:", std_score.item())
    print("Mean L2 Distance:", mean_distance.item())
    print("STD L2 Distance:", std_distance.item())
    print("Min L2 Distance:", min_distance.item())
    print("Max L2 Distance:", max_distance.item())
    stats_info = (
        f"Mean similarity score: {mean_score.item():.2f}\n"
        f"Min similarity score: {min_score.item():.2f}\n"
        f"Max similarity score: {max_score.item():.2f}\n"
        f"STD of similarity scores: {std_score.item():.2f}\n"
        f"Mean L2 Distance: {mean_distance.item():.2f}\n"
        f"STD L2 Distance: {std_distance.item():.2f}\n"
        f"Min L2 Distance: {min_distance.item():.2f}\n"
        f"Max L2 Distance: {max_distance.item():.2f}"
    )

    plot_embeddings(video_embeddings, text_embeddings, mappings,
                    'plots', 'pca_plot_nomlp2D.png', True)
    plot_embeddings_3d(video_embeddings, text_embeddings, mappings,
                       'plots', 'pca_plot_nomlp3D.png', True)
    adjust_video_embeddings = mlp_eval(video_embeddings)
    plot_embeddings(adjust_video_embeddings, text_embeddings, mappings,
                    'plots', 'pca_plot_mlp2D.png', True)
    plot_embeddings_3d(adjust_video_embeddings, text_embeddings, mappings,
                       'plots', 'pca_plot_mlp3D.png', True)