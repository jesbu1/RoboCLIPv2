import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding
import torch as th
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D
from sklearn.decomposition import PCA
import joblib
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

def reduce_dimension(video_embeddings, text_embeddings, dimension):
    video_embeddings_np = video_embeddings.numpy()
    text_embeddings_np = text_embeddings.numpy()

    combined_embeddings = np.vstack((video_embeddings_np, text_embeddings_np))
    pca = PCA(n_components=dimension)
    combined_embeddings_pca = pca.fit_transform(combined_embeddings)

    video_embeddings_pca = combined_embeddings_pca[:len(video_embeddings_np)]
    text_embeddings_pca = combined_embeddings_pca[len(video_embeddings_np):]

    video_embeddings_pca = torch.from_numpy(video_embeddings_pca).float()
    text_embeddings_pca = torch.from_numpy(text_embeddings_pca).float()
    joblib.dump(pca, f'saved_model/pca_model_{dimension}.pkl')
    return video_embeddings_pca, text_embeddings_pca

def mlp_train(num_epochs, video_embeddings, text_embeddings, dimension):
    # L1 距离
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model = SimpleMLP(input_dim=dimension, output_dim=dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    checkpoint_interval = 100
    save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model"
    if dimension != 512:
        video_embeddings, text_embeddings = reduce_dimension(video_embeddings,text_embeddings,dimension)
    for epoch in range(num_epochs):
        adjusted_video_embeddings = model(video_embeddings)
        loss = criterion(adjusted_video_embeddings, text_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'{save_dir}/checkpoint_{epoch + 1}_{dimension}.pth')
            print(f'Checkpoint saved at epoch {epoch + 1}')
    final_model_path = f"{save_dir}/final_model_{dimension}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')


def mlp_eval(video_embeddings, text_embeddings = None):

    model = SimpleMLP(input_dim=512, output_dim=512)
    model_path = "/scr/yusenluo/RoboCLIP/visualization/saved_model/final_model.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        adjusted_video_embeddings = model(video_embeddings)
        # l2_distances = torch.norm(adjusted_video_embeddings - text_embeddings, p=2, dim=1)
        # mean_distance = torch.mean(l2_distances)
        # print("Mean L2 Distance for evaluation:", mean_distance.item())
    return adjusted_video_embeddings

if __name__ == '__main__':

    video_paths = list_webm_files('../20bn-something-something-v2') #'../20bn-something-something-v2'
    video_text_dataset = VideoTextDataset(video_paths)
    data_loader = DataLoader(video_text_dataset, batch_size=4, shuffle=True, num_workers=2)

    s3d = S3D('../s3d_dict.npy', 512)
    s3d.load_state_dict(th.load('../s3d_howto100m.pth'))
    s3d.eval()
    video_embeddings, text_embeddings, embeddings_dataset, mappings = Embedding(s3d, data_loader)
    #video_embeddings, text_embeddings = reduce_dimension(video_embeddings,text_embeddings, 16)
    mlp_train(2000, video_embeddings, text_embeddings, 512)