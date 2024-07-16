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
from dataloader import GifDataset, GifProgressDataset
import torch.nn as nn
import torch as th
import wandb
import random



class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

        # self.linear_2 = nn.Linear(output_dim, output_dim)
        # self.linear_3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        # x = F.relu(x)
        # x = self.linear_2(x)
        # x = F.relu(x)
        # x = self.linear_3(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)

def triplet_loss(gt, positive, negative, type, margin = (0.1, 0.25, 0.3, 0.9), progress = None): 
    # 

    pos_sim = cosine_similarity(gt, positive)
    neg_sim = cosine_similarity(gt, negative)

    loss = torch.zeros_like(pos_sim).cuda()

    # type1: hard negative margin = 0.1
    mask_type_1 = (type == 1)
    loss[mask_type_1] = F.relu(margin[0] - pos_sim[mask_type_1] + neg_sim[mask_type_1])

    # type2: semi-hard negative margin = 0.25
    mask_type_2 = (type == 2)
    loss[mask_type_2] = F.relu(margin[1] - pos_sim[mask_type_2] + neg_sim[mask_type_2])

    # type3: adaptive margin
    mask_type_3 = (type == 3)
    progress = progress[mask_type_3]
    # adaptive margin range from 0.3 to 0.8
    adaptive_margin = (margin[2] + (margin[3] - margin[2]) * progress).cuda()
    loss[mask_type_3] = F.relu(adaptive_margin - pos_sim[mask_type_3] + neg_sim[mask_type_3])

    return loss.mean()

def triplet_loss_lower_bound(gt, positive, negative, type, margin = (0.1, 0.2, 0.3, 0.9), progress = None): # may need a lower bound for progress
    # for progress, we need to have an lower bound

    pos_sim = cosine_similarity(gt, positive)
    neg_sim = cosine_similarity(gt, negative)

    loss = torch.zeros_like(pos_sim).cuda()

    # type1: hard negative margin = 0.1
    mask_type_1 = (type == 1)
    loss[mask_type_1] = F.relu(margin[0] - pos_sim[mask_type_1] + neg_sim[mask_type_1])

    # type2: semi-hard negative margin = 0.2
    mask_type_2 = (type == 2)
    loss[mask_type_2] = F.relu(margin[1] - pos_sim[mask_type_2] + neg_sim[mask_type_2])

    # type3: adaptive margin # can change to L1 loss
    mask_type_3 = (type == 3)
    progress = progress[mask_type_3]
    # adaptive margin range from 0.3 to 0.9
    # adaptive_margin_upper_bound = (margin[2] + (margin[3] - margin[2]) * progress).cuda()
    # adaptive_margin_lower_bound = (margin[2] + (margin[3] - margin[2]) * progress - 0.1).cuda()
    # #make sure negative similarity - positive similarity is between upper and lower bound
    # simi_diff = neg_sim[mask_type_3] - pos_sim[mask_type_3]
    # loss_upper = F.relu(adaptive_margin_upper_bound - simi_diff)
    # loss_lower = F.relu(simi_diff - adaptive_margin_lower_bound)
    # loss[mask_type_3] = loss_upper + loss_lower
    progress_range = ((margin[3] - margin[2]) * progress + margin[2]).cuda()
    loss[mask_type_3] = F.l1_loss(neg_sim[mask_type_3] - pos_sim[mask_type_3], progress_range)



    return loss.mean()



def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()


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
    experiment_name = "triplet_loss" + "_" + str(args.seed) + "_" + args.model_name + "_l1"
    if args.time_shuffle:
        experiment_name += "_TimeShuffle"
    if args.time_shorten:
        experiment_name += "_TimeShort"
    if args.norm:
        experiment_name += "_Norm"
    if args.add_lower_bound:
        experiment_name += "_LowerBound"

    experiment_name += "_DoorOverFit"
    

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="adaptive_triplet_loss_training",
        config=args,
        name=experiment_name,
    )

    if model_name == "xclip":
        xclip_net = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").cuda()
        xclip_net.eval()
        # pixel_values = self.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)
        xclip_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
    else:
        s3d_model = S3D('../s3d_dict.npy', 512)
        s3d_model.load_state_dict(th.load('../s3d_howto100m.pth'))
        s3d_model.eval().cuda()
    transform_model = SingleLayerMLP(512, 512, normalize=args.norm).cuda()
    dataset = GifProgressDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # loss_func = nn.TripletMarginLoss(margin=0.5, p=1)
    optimizer = th.optim.Adam(transform_model.parameters(), lr=1e-4)



    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            
            gt_array = batch["gt_array"]
            pos_array = batch["pos_array"]
            neg_array = batch["neg_array"]
            batch_size = neg_array.shape[0]
            samples = torch.cat([gt_array, pos_array, neg_array]).cuda()
            type = batch["type"]
            progress = batch["progress"].float().cuda()
            
            with th.no_grad():
                if model_name == "xclip":
                    import pdb ; pdb.set_trace()
                    pixel_values = xclip_processor.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)


                    video_features = xclip_net.get_video_features(samples)

                else:
                    video_features = s3d_model(samples)["video_embedding"]
            video_features = normalize_embeddings(video_features)
            video_features = transform_model(video_features)
            gt_features = video_features[:batch_size]
            pos_features = video_features[batch_size:2*batch_size]
            neg_features = video_features[2*batch_size:]
            if args.add_lower_bound:
                loss = triplet_loss_lower_bound(gt_features, pos_features, neg_features, type, progress=progress)
            else:
                loss = triplet_loss(gt_features, pos_features, neg_features, type, progress=progress)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {"loss": loss.item()}
            print(wandb_log)
            wandb.log(wandb_log)
        
        if epoch % 1 == 0:
            if args.model_name == "xclip":
                if not os.path.exists(f"/home/jzhang96/triplet_loss_models/{experiment_name}"):
                    os.makedirs(f"/home/jzhang96/triplet_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(), f"/home/jzhang96/triplet_loss_models/{experiment_name}/{epoch}.pth")
            else:
                if not os.path.exists(f"/home/jzhang96/triplet_loss_models/{experiment_name}"):
                    os.makedirs(f"/home/jzhang96/triplet_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(), f"/home/jzhang96/triplet_loss_models/{experiment_name}/{epoch}.pth")
            

            
                
                



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='xclip', choices=['xclip', 's3d'])
    argparser.add_argument('--time_shuffle', action='store_true')
    argparser.add_argument('--h5_path', type=str, default='/home/jzhang96/metaworld_gifs_1.h5')
    argparser.add_argument('--time_shorten', action='store_true')
    argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--add_lower_bound', action='store_true')
    args = argparser.parse_args()
    main(args)