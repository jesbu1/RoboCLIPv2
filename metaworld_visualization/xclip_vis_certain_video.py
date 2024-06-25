import os
import sys
# add the "../" directory to the sys.path
parent_dir_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
parent_dir_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xclip/'))
sys.path.append(parent_dir_1)
sys.path.append(parent_dir_2)
import cv2
import random
from tqdm import tqdm
# import xclip
from pca_utils import normalize_embeddings
from transformers import AutoTokenizer, AutoModel, AutoProcessor 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import numpy as np


choose_task = [13,14,15,23,24,25,35,36,37]
video_sample_per_task = 10



def main(): 
    '''
    Main function.
    '''
    # h5_file_path = "/scr/jzhang96/metaworld_xclip_embeddings.h5"
    h5_file_path = "/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5"
    h5_file = h5py.File(h5_file_path, 'r')
    
    total_video_features = []
    total_task_name = []
    for task in choose_task:
        single_task = str(task)
        data_group = h5_file[single_task]
        task_name = data_group.attrs["task_name"].split("-v2")[0]

        video_idx = len(list(data_group.keys()))
        choose_idx_range = random.sample(range(video_idx), video_sample_per_task)
        print("choose_idx: ", choose_idx_range)
        this_video_feature = []
        for idx in choose_idx_range:
            video_feature = data_group[str(idx)]["xclip_video_feature"]
            this_video_feature.append(video_feature)
        this_video_feature = np.array(this_video_feature)
        total_video_features.append(this_video_feature)
        total_task_name.append(task_name)
    h5_file.close()
    total_video_features = np.concatenate(total_video_features, axis=0)
    total_video_features = normalize_embeddings(total_video_features)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(total_video_features)
    print("reduced_embeddings: ", reduced_embeddings.shape)


    # Plot the reduced embeddings
    colors = plt.cm.tab10(np.linspace(0, 1, len(choose_task)))  # 28 groups

    for i in range(len(choose_task)):
        group_data = reduced_embeddings[i*video_sample_per_task:(i+1)*video_sample_per_task]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = total_task_name[i]
        plt.scatter(x, y, color=colors[i], label=text_name)
    
    plt.title('Different Scenes PCA Visualization')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')

    plt.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 1, 1]) # adjust the plot to the right (to fit the legend

    # plt.legend()
    # save png
    plt.savefig('pca_metaworld_part_videos_same_diff.png')


            





    # total_video_features = np.array(total_video_features)
    # total_video_features = normalize_embeddings(total_video_features)
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(total_video_features)


    # # Plot the reduced embeddings
    # colors = plt.cm.tab10(np.linspace(0, 1, true_folders))  # 28 groups

    # for i in range(true_folders):
    #     group_data = reduced_embeddings[i*10:(i+1)*10]
    #     x = group_data[:, 0]
    #     y = group_data[:, 1]
    #     text_name = text[i]
    #     plt.scatter(x, y, color=colors[i], label=text_name)
    
    # plt.title('2D PCA for Metaworld Success Videos')
    # plt.xlabel('x-dim')
    # plt.ylabel('y-dim')

    # plt.legend(loc='upper left', ncol=4)
    # plt.tight_layout(rect=[0, 0, 0.5, 1]) # adjust the plot to the right (to fit the legend

    # # plt.legend()
    # # save png
    # plt.savefig('pca_metaworld_success_videos.png')


    # # print("Total number of tasks with enough videos: ", num)

if __name__ == "__main__":
    main()