import torch    
from s3dg import S3D
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import argparse
import h5py
from tqdm import tqdm
import os
import imageio
from torch.utils.data import DataLoader
# from dataloader import GifDataset, GifProgressDataset, GifProgressTrainDataset
import torch.nn as nn
import torch as th
import wandb
import random
import matplotlib.pyplot as plt
import copy
from dataloader_text import GifTextDataset
from sklearn.decomposition import PCA


def plot_distribution(transform_model, evaluate_run_embeddings, total_evaluate_embeddings, evaluate_tasks, total_evaluate_tasks, eval_text_embedding):

    total_evaluate_embeddings = transform_model(total_evaluate_embeddings.clone()).detach().cpu().numpy()
    evaluate_run_embeddings = transform_model(evaluate_run_embeddings.clone()).detach().cpu().numpy()
    eval_text_embedding = normalize_embeddings(eval_text_embedding, False)
    evaluate_run_embeddings = normalize_embeddings(evaluate_run_embeddings, False)
    total_evaluate_embeddings = normalize_embeddings(total_evaluate_embeddings, False)


    total_embedding = np.concatenate([total_evaluate_embeddings, eval_text_embedding], axis=0)
    pca = PCA(n_components=2)
    pca_model = pca.fit(total_embedding)
    total_video_embedding = pca_model.transform(total_evaluate_embeddings)
    run_video_embedding = pca_model.transform(evaluate_run_embeddings)
    eval_text_embedding = pca_model.transform(eval_text_embedding)

    figure_1 = plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(total_evaluate_tasks)))  
    for i in range(len(total_evaluate_tasks)):
        group_data = total_video_embedding[i*10:(i+1)*10]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = total_evaluate_tasks[i]
        plt.scatter(x, y, color=colors[i], label=text_name)
    
    plt.title('2D PCA for Metaworld Total Videos')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')
    # plt.legend(loc='upper left', ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 1]) # adjust the plot to the right (to fit the legend)
    

    figure_2 = plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(evaluate_tasks)))
    for i in range(len(evaluate_tasks)):
        group_data = run_video_embedding[i*10:(i+1)*10]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = evaluate_tasks[i].split("-v2")[0]
        text_embedding = eval_text_embedding[i]
        plt.scatter(x, y, color=colors[i], label=text_name, marker='o', s=100, zorder=2)
        # put "x" above the point
        plt.scatter(text_embedding[0], text_embedding[1], color=colors[i], marker='x', s=100, zorder=3)

    plt.title('2D PCA for Metaworld Evaluate Videos')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')
    plt.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    # plt.tight_layout(rect=[0, 0, 0.8, 1]) # adjust the plot to the right (to fit the legend)
    plt.tight_layout() # adjust the plot to the right (to fit the legend)

    return figure_1, figure_2


class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


if __name__ == '__main__':

    evaluate_h5 = h5py.File("metaworld_s3d_embedding.h5", "r")
    evaluate_tasks = ["door-close-v2-goal-hidden", "door-open-v2-goal-hidden", "drawer-close-v2-goal-hidden", "button-press-v2-goal-hidden", "button-press-topdown-v2-goal-hidden"]
    total_evaluate_tasks = list(evaluate_h5.keys())
    total_evaluate_embeddings = None
    evaluate_run_embeddings = None
    for keys in evaluate_h5.keys():
        task_data = np.asarray(evaluate_h5[keys])
        # random choose 10
        choose_index = np.random.choice(task_data.shape[0], 10, replace=False)
        task_data = task_data[choose_index]
        if total_evaluate_embeddings is None:
            total_evaluate_embeddings = task_data
        else:
            total_evaluate_embeddings = np.concatenate([total_evaluate_embeddings, task_data], axis=0)

        if keys in evaluate_tasks:
            if evaluate_run_embeddings is None:
                evaluate_run_embeddings = task_data
            else:
                evaluate_run_embeddings = np.concatenate([evaluate_run_embeddings, task_data], axis=0)

    total_evaluate_embeddings = torch.tensor(total_evaluate_embeddings).cuda()
    evaluate_run_embeddings = torch.tensor(evaluate_run_embeddings).cuda()
    evaluate_h5.close()

    text_h5 = h5py.File("metaworld_s3d_text.h5", "r")
    eval_text_embedding = []
    for keys in evaluate_tasks:
        embedding = np.asarray(text_h5[keys]["embedding"])
        embedding = np.expand_dims(embedding, axis=0)
        eval_text_embedding.append(embedding)
    eval_text_embedding = np.concatenate(eval_text_embedding, axis=0)
    eval_text_embedding = normalize_embeddings(eval_text_embedding)
    
    text_h5.close()

    transform_model = SingleLayerMLP(512, 512).cuda()
    transform_model.load_state_dict(th.load("/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/triplet_loss_42_s3d_l1_0.9_bs_675_Norm_text_8_24_38_41_47/999.pth"))

    figure_1, figire_2 = plot_distribution(transform_model, 
                                            evaluate_run_embeddings, 
                                            total_evaluate_embeddings, 
                                            evaluate_tasks, 
                                            total_evaluate_tasks, 
                                            eval_text_embedding)
    figure_1.savefig('figure_1.png', format='png', dpi=100) 
    figire_2.savefig('figire_2.png', format='png', dpi=100) 

    # total_evaluate_embeddings = transform_model(total_evaluate_embeddings)

