import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding, filter_top_embeddings, SthDataset, OpenXDataset
from sklearn.model_selection import train_test_split
import torch as th
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D
from sklearn.decomposition import PCA
import joblib
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mlp import normalize_embeddings, standradize_embeddings
import argparse


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class A_Model(nn.Module):
    def __init__(self, X_S, X_T):
        super(A_Model, self).__init__()
        if isinstance(X_S, np.ndarray):
            X_S = torch.from_numpy(X_S).float()
        if isinstance(X_T, np.ndarray):
            X_T = torch.from_numpy(X_T).float()
        self.X_S = nn.Parameter(X_S)
        self.X_T = nn.Parameter(X_T)

    def forward(self):
        A = torch.matmul(self.X_T.t(), self.X_T)
        A = torch.matmul(A, torch.matmul(self.X_S.t(), self.X_S))
        return A


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)


def reduce_dimension(
    embeddings, variance_threshold, train_size, embed_type, dimension=None, filter = False
):
    if dimension:
        pca = PCA(n_components=dimension)
    else:
        pca = PCA(n_components=variance_threshold)
    reduced_embeddings = pca.fit_transform(embeddings)
    if filter:
        model_filename = (
            f"saved_model/M/OpenX/droid/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"saved_model/M/OpenX/droid/pca_model/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return pca.components_ , torch.from_numpy(reduced_embeddings).float()


def reduce_dimension_trained(
    embeddings, variance_threshold, train_size, embed_type, filter = False
):
    if filter:
        model_filename = (
            f"saved_model/M/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"saved_model/M/OpenX/droid/pca_model/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    pca = joblib.load(model_filename)
    reduced_embeddings = pca.transform(embeddings)

    print(f"Using PCA_{embed_type} from {model_filename}")
    return pca.components_ , torch.from_numpy(reduced_embeddings).float()


def compute_M(video_embeddings, text_embeddings, variance_threshold, train_size, filter):
    
    M = np.dot(X_S, X_T.T) # 35 35
    print("X_S", np.dot(X_S, X_S.T))
    print("X_T", np.dot(X_T, X_T.T))
    M_tensor = torch.from_numpy(M).float()
    # A_1 = np.dot(X_S.T, X_S)
    # A_2 = np.dot(X_T.T, X_T)
    # A = np.dot(A_2, A_1) # 512,512
    # A_tensor = torch.from_numpy(A).float()
    if filter:
        save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/filter"
    else:
        save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/alignXtoX"
    M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
    torch.save(M_tensor, M_model_path)
    print(f'A model saved to {M_model_path}')
    return M_tensor

# def compute_M_1(video_embeddings, text_embeddings, variance_threshold, train_size):
#     video_embeddings = normalize_embeddings(video_embeddings)
#     text_embeddings = normalize_embeddings(text_embeddings)
#     X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', 512) # 35, 512
#     X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video', X_T.shape[0])# 35，512
#     M = np.dot(X_S, X_T.T) # 35 35
#     A = np.dot(X_S.T, M) # 512 35
#     A_tensor = torch.from_numpy(A).float()
#
#     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M_1/filter"
#     M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
#     torch.save(A_tensor, M_model_path)
#     print(f'M model saved to {M_model_path}')
#
# def compute_M_2(video_embeddings, text_embeddings, variance_threshold, train_size):
#     video_embeddings = normalize_embeddings(video_embeddings)
#     text_embeddings = normalize_embeddings(text_embeddings)
#     X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', 512) # 35, 512
#     X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video', X_T.shape[0])# 35，512
#     M = np.dot(X_S.T, X_T) # 512 512
#     A = np.dot(X_S, M) # 35 512
#     A = A.T #512 35
#     A_tensor = torch.from_numpy(A).float()
#
#     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M_2/filter"
#     M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
#     torch.save(A_tensor, M_model_path)
#     print(f'M model saved to {M_model_path}')
#
# def compute_M_3(video_embeddings, text_embeddings, variance_threshold, train_size):
#     video_embeddings = normalize_embeddings(video_embeddings)
#     text_embeddings = normalize_embeddings(text_embeddings)
#     X_T, reduced_text = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text', 512) # 35, 512
#     X_S, reduced_video = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video', X_T.shape[0])# 35，512
#     M = np.dot(X_S, X_S.T) # 35 35
#     A = np.dot(M, X_T) # 35 512
#     A = A.T #512 35
#     A_tensor = torch.from_numpy(A).float()
#     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M_3/filter"
#     M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}.pth"
#     torch.save(A_tensor, M_model_path)
#     print(f'M model saved to {M_model_path}')


def eval_M(video_embeddings_pca, M_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    M_model = nn.Linear(video_embeddings_pca.shape[1], video_embeddings_pca.shape[1], bias=False).to(device)
    M_model.load_state_dict(torch.load(M_path, map_location=torch.device(device)))
    M_model.eval()
    #Matrix = torch.load(M_path).to(device)
    with torch.no_grad():
        #adjust_video_embeddings = torch.matmul(video_embeddings_pca, Matrix)
        adjust_video_embeddings = M_model(video_embeddings_pca)
    return adjust_video_embeddings


def eval_MLP(video_embeddings, M_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = SimpleMLP(video_embeddings.shape[1],video_embeddings.shape[1]).to(device)
    model.load_state_dict(torch.load(M_path, map_location=torch.device(device)))
    with torch.no_grad():
        adjust_video_embeddings = model(video_embeddings)
    return adjust_video_embeddings

def eval_A(text_embeddings, video_embeddings, A_path, X_S = None, X_T = None):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # A_model = nn.Linear(text_embeddings.shape[1], text_embeddings.shape[1], bias=False)
    A_model = A_Model(X_S, X_T).to(device)
    A_model.load_state_dict(torch.load(A_path, map_location=device))
    A_model.eval()

    #Matrix = torch.load(A_path)

    with torch.no_grad():
        #simi_score = torch.matmul(text_embeddings, Matrix) @ video_embeddings.T
        #simi_score = A_model(text_embeddings) @ video_embeddings.T
        simi_score = torch.matmul(text_embeddings, A_model()) @ video_embeddings.T
    return simi_score, torch.matmul(text_embeddings, A_model())


def eval_A_separate(text_embeddings, video_embeddings, A_path, X_S = None, X_T = None):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    A_model = A_Model(X_S, X_T).to(device)
    A_model.load_state_dict(torch.load(A_path, map_location=device))
    A_model.eval()
    X_S = A_model.X_S.data
    X_T = A_model.X_T.data
    M = torch.matmul(X_S, X_T.T)
    pca_v = video_embeddings @ X_S.T
    adjusted_v = pca_v @ M
    pca_t = text_embeddings @ X_T.T
    print("video norms after M model:", th.norm(adjusted_v, dim=1))
    print("text norms after pca:", th.norm(pca_t, dim=1))
    cos_simi = np.mean(np.diag
                       (cosine_similarity(adjusted_v.numpy(),
                                          pca_t.numpy())))
    with torch.no_grad():
        simi_score = pca_t @ adjusted_v.T #torch.matmul(text_embeddings, A_model()) @ video_embeddings.T
        #print('simi_matriix', simi_score)
        #print('A', A_model())
        print('cos_matrix', cosine_similarity(adjusted_v.numpy(),
                                          pca_t.numpy()))
        print("X_S", X_S @ X_S.T)
        print("X_T", X_T @ X_T.T)
    return simi_score, torch.matmul(text_embeddings, A_model()), cos_simi, adjusted_v, pca_t

def cos_similarity_score(adjust_video_embeddings, text_embeddings_pca):

    sim_scores = F.cosine_similarity(adjust_video_embeddings, text_embeddings_pca, dim=1)
    return sim_scores


def finetune_M(num_epochs, reduced_video, reduced_text, variance_threshold, current_sample_size, M=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_path = f'/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/alignXtoX/M_model_{variance_threshold}_{current_sample_size}.pth'
    
    save_dir = '/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/finetune_M'

    if M != None:
        pretrained_matrix = M
    model = nn.Linear(reduced_text.shape[1], reduced_text.shape[1], bias=False).to(device)
    #model = SimpleMLP(512, 512).to(device)

    reduced_text = reduced_text.to(device)
    reduced_video = reduced_video.to(device)
    
    model.weight.data = pretrained_matrix.T.to(device)

    # with torch.no_grad():
    #     adjust_video_embeddings_0 = torch.matmul(reduced_video, pretrained_matrix)
    #     adjusted_video_embeddings_1 = model(reduced_video)
    #     print(torch.allclose(adjust_video_embeddings_0, adjusted_video_embeddings_1)) #True
    milnce_loss = MILNCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epochs):
        adjusted_video_embeddings = model(reduced_video)
        # similarity_matrix = reduced_text @ adjusted_video_embeddings.T
        # diagonal_similarities = torch.diag(similarity_matrix)
        # loss = -torch.mean(diagonal_similarities)
        loss = milnce_loss(adjusted_video_embeddings, reduced_text)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    final_model_path = f"{save_dir}/M_model_{variance_threshold}_{current_sample_size}_milnce.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    return model


def finetune_MLP(num_epochs, reduced_video, reduced_text, variance_threshold, current_sample_size):
    save_dir = '/scr/yusenluo/RoboCLIP/visualization/saved_model/OpenX/droid/mlp_model'
    milnce_loss = MILNCELoss()

    model = SimpleMLP(reduced_video.shape[1], reduced_text.shape[1]).to(device)
    reduced_text = reduced_text.to(device)
    reduced_video = reduced_video.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epochs):
        adjusted_video_embeddings = model(reduced_video)
        # similarity_matrix = reduced_text @ adjusted_video_embeddings.T
        # diagonal_similarities = torch.diag(similarity_matrix)
        # loss = -torch.mean(diagonal_similarities)
        loss = milnce_loss(adjusted_video_embeddings, reduced_text)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    final_model_path = f"{save_dir}/MLP_model_{variance_threshold}_{current_sample_size}_milnce_3000.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

def finetune_A(num_epochs, video_embeddings, text_embeddings, variance_threshold, current_sample_size):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    video_embeddings = normalize_embeddings(video_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)
    A_path = f'/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/A_model_{variance_threshold}_{current_sample_size}.pth'
    X_T, reduced_text = reduce_dimension_trained(text_embeddings, variance_threshold, current_sample_size, 'text',
                                         filter=filter)
    X_S, reduced_video = reduce_dimension_trained(video_embeddings, variance_threshold,
                                          current_sample_size, 'video', filter=filter)  # 35，512
    save_dir = '/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/random_finetune_A'
    print("X_S", np.dot(X_S, X_S.T))
    print("X_T", np.dot(X_T, X_T.T))
    pretrained_matrix = torch.load(A_path)
    print("A", pretrained_matrix)
    #model = nn.Linear(pretrained_matrix.shape[0], pretrained_matrix.shape[1], bias=False).to(device)
    model = A_Model(X_S, X_T).to(device)

    # torch.save(model.state_dict(),
    #            f"/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/random_A/A_model_{variance_threshold}_{current_sample_size}.pth")
    #model.weight.data = pretrained_matrix.T

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epochs):
        similarity_matrix = th.matmul(text_embeddings.to(device), model()) @ video_embeddings.T.to(device) #model(text_embeddings.to(device)) @ video_embeddings.T.to(device)  ##  #batch, batch
        diagonal_similarities = torch.diag(similarity_matrix)
        loss = -torch.mean(diagonal_similarities)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    final_model_path = f"{save_dir}/A_model_{variance_threshold}_{current_sample_size}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=bool, default=False)
    args = parser.parse_args()
    filter = False
    if args.filter:
        filter = True
    variance_thresholds = [0.9, 0.95]
    sample_sizes = [10] #[10, 20, 50, 100, 200] #[1, 2, 4, 8, 16, 21]

    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths = list_webm_files(
        "../20bn-something-something-v2/train"
    )  # '../20bn-something-something-v2'
   # print(len(video_paths))
    s3d = S3D("../s3d_dict.npy", 512)
    s3d.load_state_dict(torch.load("../s3d_howto100m.pth"))
    s3d.eval()

    video_text_dataset = OpenXDataset(
        '/scr/yusenluo/RoboCLIP/OpenX/droid', random_samples=False, dataset_name='droid'
    )
    seen_labels = set()
    unique_indices = []
    for idx in range(len(video_text_dataset)):
        item = video_text_dataset[idx]
        text_label = item['text']
        if text_label not in seen_labels:
            seen_labels.add(text_label)
            unique_indices.append(idx)
    unique_dataset = Subset(video_text_dataset, unique_indices)
    train_dataset, val_dataset = train_test_split(unique_dataset, test_size=0.2, random_state=42)
    for size_multiplier in sample_sizes:
        #current_sample_size = 50 * size_multiplier
        # if current_sample_size == 1050:
        #     variance_thresholds.append(512)
        # video_text_dataset = VideoTextDataset(
        #     video_paths, num_samples=current_sample_size, random_samples=False
        # )
        # data_loader = DataLoader(
        #     video_text_dataset, batch_size=50, shuffle=False, num_workers=10
        # )
        # video_text_dataset = SthDataset(
        #         "../20bn-something-something-v2/train", random_samples=False
        #     )
        
        #torch.manual_seed(42)
        # indices = torch.randperm(len(train_dataset))[:size_multiplier]
        # limited_dataset = Subset(train_dataset, indices)
        # current_sample_size = len(limited_dataset)
        current_sample_size = len(train_dataset)
        data_loader = DataLoader(
            train_dataset, batch_size=50, shuffle=False, num_workers=5
        )
        video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(
            s3d, data_loader
        )
        '''
        test
        '''
        #video_embeddings = 2 * text_embeddings - 0.5
        if filter:
            video_embeddings, text_embeddings = filter_top_embeddings(video_embeddings, text_embeddings, 0.5)

        video_embeddings_normalized = normalize_embeddings(
            video_embeddings
        ).clone()
        text_embeddings_normalized = normalize_embeddings(
            text_embeddings
        ).clone()
        for variance_threshold in variance_thresholds:
            print(
                f"Training with variance threshold {variance_threshold} and sample size {current_sample_size}."
            )
            if variance_threshold == 0:
                reduced_text = text_embeddings_normalized.float()
                reduced_video = video_embeddings_normalized.float()
            else:
                X_T, reduced_text = reduce_dimension(text_embeddings_normalized, variance_threshold, current_sample_size,
                                                             'text',
                                                             filter=filter)
                X_S, reduced_video = reduce_dimension(video_embeddings_normalized, variance_threshold,
                                                              current_sample_size, 'video', filter=filter, dimension=reduced_text.shape[1])
            
            M = compute_M(X_S, X_T, variance_threshold, current_sample_size, filter)
            finetuned_M = finetune_M(3000, video_embeddings_normalized, text_embeddings_normalized, variance_threshold, current_sample_size, M)
            # finetune_MLP(3000, reduced_video, reduced_text, variance_threshold,
            #                          current_sample_size)
            
