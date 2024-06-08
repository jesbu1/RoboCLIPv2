import torch
import numpy as np
import ot



def get_distribution(embeddings):
    '''
    return a gaussian distribution of the embeddings

    Input: embeddings: np.array of shape (n, d)
    '''
    # get the mean and covariance of the embeddings
    embeddings = torch.tensor(embeddings)
    mean = embeddings.mean(dim=0)
    cov = torch.mm(embeddings.T, embeddings) / embeddings.shape[0]
    dist = torch.distributions.MultivariateNormal(mean, cov)
    return dist
