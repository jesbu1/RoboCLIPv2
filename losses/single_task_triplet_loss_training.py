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
from dataloader import GifDataset
import torch.nn as nn
import torch as th
import wandb
import random



class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x




def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    h5_file = h5py.File(args.h5_path, "r")
    model_name = args.model_name

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    # wandb_eval_task_name = "_".join([str(i) for i in eval_tasks])
    # experiment_name = args.experiment_name + "_" + wandb_eval_task_name + "_" + str(args.seed)
    # if args.mse:
    #     experiment_name = experiment_name + "_mse_" + str(args.mse_weight)
    experiment_name = "triplet_loss" + "_" + str(args.seed) + "_" + args.model_name
    if args.time_shuffle:
        experiment_name += "_TimeShuffle"
    if args.time_shorten:
        experiment_name += "_TimeShort"
    if args.norm:
        experiment_name += "_Norm"
    

    # if args.wandb:
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="triplet_loss",
        config=args,
        name=experiment_name,
    )

    if model_name == "xclip":
        xclip_net = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").cuda()
        xclip_net.eval()
    else:
        s3d_model = S3D()
        s3d_model.load_state_dict(th.load('s3d_howto100m')).cuda()
        s3d_model.eval()
    transform_model = SingleLayerMLP(512, 512, normalize=args.norm).cuda()
    dataset = GifDataset(args)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
    epochs = 10

    loss_func = nn.TripletMarginLoss(margin=0.5, p=1)
    optimizer = th.optim.Adam(transform_model.parameters(), lr=1e-4)



    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            
            gt_array = batch["gt_array"]
            pos_array = batch["pos_array"]
            neg_array = batch["neg_array"]
            batch_size = neg_array.shape[0]
            samples = torch.cat([gt_array, pos_array, neg_array]).cuda()
            
            with th.no_grad():
                if model_name == "xclip":
                    video_features = xclip_net.get_video_features(samples)
                    import pdb; pdb.set_trace()

                else:
                    video_features = s3d_model(samples)["video_embedding"]
                    import pdb; pdb.set_trace()

            video_features = transform_model(video_features)
            gt_features = video_features[:batch_size]
            pos_features = video_features[batch_size:2*batch_size]
            neg_features = video_features[2*batch_size:]

            loss = loss_func(gt_features, pos_features, neg_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {"loss": loss.item()}
            wandb.log(wandb_log)
        
        if epoch % 10 == 0:
            th.save(transform_model.state_dict(), f"/scr/jzhang96/triplet_loss_models/transform_model_{epoch}.pth")

        



            # Calculate triplet loss


            

            
                
                


 


            import pdb; pdb.set_trace()







if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='xclip', choices=['xclip', 's3d'])
    argparser.add_argument('--time_shuffle', action='store_true')
    argparser.add_argument('--h5_path', type=str, default='/scr/jzhang96/metaworld_gifs_1.h5')
    argparser.add_argument('--time_shorten', action='store_true')
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    args = argparser.parse_args()
    main(args)
