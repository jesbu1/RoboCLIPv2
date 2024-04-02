import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding
import torch as th
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D
from sklearn.decomposition import PCA
import joblib
import torch.nn.functional as F
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

def normalize_embeddings(embeddings):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings

def reduce_dimension(embeddings, variance_threshold, train_size, embed_type):
    normalized_embeddings = normalize_embeddings(embeddings)
    pca = PCA(n_components=variance_threshold)
    reduced_embeddings = pca.fit_transform(normalized_embeddings)
    model_filename = f'saved_model/pca_model_{embed_type}_{variance_threshold}_{train_size}_nonorm.pkl'
    joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return torch.from_numpy(reduced_embeddings).float()


def mlp_train(num_epochs, video_embeddings, text_embeddings, variance_threshold, train_size):
    video_embeddings = normalize_embeddings(video_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)
    video_embeddings = reduce_dimension(video_embeddings, variance_threshold, train_size, 'video')
    text_embeddings = reduce_dimension(text_embeddings, variance_threshold, train_size, 'text')
    video_embeddings = video_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)

    # L1 Distance
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model = SimpleMLP(input_dim=video_embeddings.shape[1], output_dim=text_embeddings.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model"
    for epoch in range(num_epochs):
        adjusted_video_embeddings = model(video_embeddings)
        loss = criterion(adjusted_video_embeddings, text_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        #checkpoint_interval = 100
        # if (epoch + 1) % checkpoint_interval == 0:
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     torch.save(checkpoint, f'{save_dir}/checkpoint_{epoch + 1}_{dimension}_norm.pth')
        #     print(f'Checkpoint saved at epoch {epoch + 1}')
    final_model_path = f"{save_dir}/final_model_{variance_threshold}_{train_size}_nonorm.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')


def mlp_eval(video_embeddings, text_embeddings, model_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim=video_embeddings.shape[1], output_dim=text_embeddings.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path))
    #print(video_embeddings.shape[1], text_embeddings.shape[1])
    model.eval()

    with torch.no_grad():
        adjusted_video_embeddings = model(video_embeddings)
        # l2_distances = torch.norm(adjusted_video_embeddings - text_embeddings, p=2, dim=1)
        # mean_distance = torch.mean(l2_distances)
        # print("Mean L2 Distance for evaluation:", mean_distance.item())
    return adjusted_video_embeddings


if __name__ == '__main__':
    variance_thresholds = [0.9, 0.95, 0.99]
    sample_sizes = [1, 2, 4, 8, 16]
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths = list_webm_files('../20bn-something-something-v2/train')  # '../20bn-something-something-v2'
    s3d = S3D('../s3d_dict.npy', 512)
    s3d.load_state_dict(torch.load('../s3d_howto100m.pth'))
    s3d.eval()

    for variance_threshold in variance_thresholds:
        for size_multiplier in sample_sizes:
            current_sample_size = 50 * size_multiplier
            video_text_dataset = VideoTextDataset(video_paths, num_samples=current_sample_size, random_samples=False)
            data_loader = DataLoader(video_text_dataset, batch_size=50, shuffle=True, num_workers=2)
            video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(s3d, data_loader)

            print(f'Training with variance threshold {variance_threshold} and sample size {current_sample_size}.')

            mlp_train(num_epochs=2000, video_embeddings=video_embeddings, text_embeddings=text_embeddings,
                      variance_threshold=variance_threshold, train_size=current_sample_size)