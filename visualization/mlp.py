import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding
import torch as th
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

def mlp_train(num_epochs, video_embeddings, text_embeddings):
    # L1 距离
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model = SimpleMLP(input_dim=512, output_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    checkpoint_interval = 100
    save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model"
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
            torch.save(checkpoint, f'{save_dir}/checkpoint_{epoch + 1}_norm.pth')
            print(f'Checkpoint saved at epoch {epoch + 1}')
    final_model_path = f"{save_dir}/final_model_norm.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')


def mlp_eval(video_embeddings, text_embeddings = None):

    model = SimpleMLP(input_dim=512, output_dim=512)
    model_path = "/scr/yusenluo/RoboCLIP/visualization/saved_model/final_model_norm.pth"
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
    mlp_train(2000, video_embeddings, text_embeddings)