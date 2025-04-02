from models_pe import ClassProgressTransformer
import os 
import torch
import h5py
# Load the model
model_path = os.path.join('/home/jzhang96/RoboCLIPv2/real_robot_exp','models', 'real_world_PosEmb_View_top_Rewind_ratio_0.8_EMA_momentum_0.3_End_Rewind_ratio_0.1', 'model_23.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_dim = 768
text_dim = 384

model_dict = torch.load(model_path)
args = model_dict['args']
model = ClassProgressTransformer(
        args=args,
        video_dim=video_dim,  # Original video embedding dimension
        text_dim=text_dim,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)

# load original model
model.load_state_dict(model_dict['model'])
model.eval()

# load ema model
model.load_state_dict(model_dict['ema_model'])
model.eval()

traj_data = torch.rand(1, 16, 768).to(device)
text_embeddings = torch.rand(1, 384).to(device)
pred_reward, _ = model(traj_data, text_embeddings)
pred_reward = pred_reward.squeeze(-1) # reward shape (1, 16, 1) -> (1, 16)
pred_reward = pred_reward[:, 1:] # remove the first element
print(pred_reward.shape) # shape: [1, 15]

h5_file = h5py.File('/home/jzhang96/RoboCLIPv2/real_robot_exp/usc_koch_rewind_dino_reward_main_train.h5', 'r')

traj_data = h5_file[list(h5_file.keys())[0]]
video_emb = torch.tensor(traj_data['8']).to(device)
text_emb = torch.tensor(traj_data["minilm_lang_embedding"])[0:1].to(device)
video_emb = video_emb[::2].unsqueeze(0)
# video_emb = video_emb[:].unsqueeze(0)

pred_reward, _ = model(video_emb, text_emb)
print(pred_reward)



