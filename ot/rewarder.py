import numpy as np
import torch
import ot
import numpy as np

'''
usage example:
			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
'''



def optimal_transport_plan(X,
                           Y,
                           cost_matrix,
                           method='sinkhorn_gpu',
                           niter=500,
                           epsilon=0.01):
    # X is the source distribution, Y is the target distribution
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    c_m = cost_matrix.data.detach().cpu().numpy()
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan


def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c