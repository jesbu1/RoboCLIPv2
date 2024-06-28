from sklearn.decomposition import PCA
# from torch.utils.data import Dataset
# from Transformation_Matrix import MILNCELoss
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F




def learn_pca_model(text_embedding, video_embedding, n_components = 512):

    pca_text_model = PCA(n_components=n_components)
    pca_video_model = PCA(n_components=n_components)

    pca_text_model.fit(text_embedding)
    pca_video_model.fit(video_embedding)

    return pca_text_model, pca_video_model


def transform_embeddings(pca_model, embedding):

    return pca_model.transform(embedding)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=False):
        super(MLP, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)  # First fully connected layer
        self.normalize = normalize

    def forward(self, x):
        x = self.fc(x)  # Output layer
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)  # L2 normalize the output
        return x


def get_model(dim = 512, normalize = False):
    model = MLP(dim, dim, normalize)
    return model


def compute_M(pca_text_model, pca_video_model):
    # transform the video features to the text feature space not the text feature to the video feature space
    M = np.dot(pca_video_model.components_, pca_text_model.components_.T)
    M_tensor = torch.from_numpy(M).float()
    return M_tensor
    

class MILNCELoss(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)


def plot_scatter_text(video_embeddings, text_embeddings, eval_task_name, sample_per_task):
    '''
    plot 2D scatter plot of video embeddings in 2D with different colors for different tasks
    '''
    pca = PCA(n_components=2)
    text_embeddings = text_embeddings[::sample_per_task]
    total_embeddings = np.concatenate((text_embeddings, video_embeddings), axis=0)
    total_embeddings = pca.fit_transform(total_embeddings)

    text_embeddings = total_embeddings[:len(eval_task_name)]
    video_embeddings = total_embeddings[len(eval_task_name):]
    figure = plt.figure()
    # Plot the reduced embeddings
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_task_name)))  # 28 groups

    for i in range(len(eval_task_name)):
        group_data = video_embeddings[i*sample_per_task:(i+1)*sample_per_task]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = eval_task_name[i]
        # use "." to represent the text embeddings
        plt.scatter(x, y, color=colors[i], label=text_name, marker='.', zorder=1)

        text_data = text_embeddings[i]
        x = text_data[0]
        y = text_data[1]
        plt.scatter(x, y, color=colors[i], marker='+', zorder=2)
    
    
    plt.title('Different Scenes PCA Visualization')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')

    plt.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 1, 1]) # adjust the plot to the right (to fit the legend

    return figure
    

def compute_similarity_sorted(video_embeddings, text_embeddings, task_names, video_sample_per_task):
    '''
    compute the similarity between video embeddings and text embeddings
    '''
    text_single_embeddings = text_embeddings[::video_sample_per_task]
    similarity = np.dot(video_embeddings, text_single_embeddings.T)

    # sort the similarity matrix, from high to low
    similarity_sorted = np.argsort(similarity, axis=1)[:, ::-1][:, :5]

    top1_accuracy = 0
    top3_accuracy = 0
    top5_accuracy = 0
    GT_label = np.zeros(len(video_embeddings))
    for i in range(len(task_names)):
        GT_label[i*video_sample_per_task:(i+1)*video_sample_per_task] = i
    
    for i in range(len(GT_label)):
        label = GT_label[i]
        if label in similarity_sorted[i][:1]:
            top1_accuracy += 1
        if label in similarity_sorted[i][:3]:
            top3_accuracy += 1
        if label in similarity_sorted[i][:5]:
            top5_accuracy += 1

    top1_accuracy = top1_accuracy / len(GT_label)
    top3_accuracy = top3_accuracy / len(GT_label)
    top5_accuracy = top5_accuracy / len(GT_label)
    return top1_accuracy, top3_accuracy, top5_accuracy







