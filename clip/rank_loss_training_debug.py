import torch
from PIL import Image
from dataloader_clipliv import ClipLivDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, SingleLayerMLP, ThreeLayerMLP
from torch.nn import MarginRankingLoss
import wandb
from tqdm import tqdm
import h5py

def RankLoss(
        transform_model,
        text_embedding, 
        first_frame_embedding, 
        last_frame_embedding, 
        mid_frames_1_embedding, 
        mid_frames_2_embedding, 
        other_video_frame_embedding,
        soft_margin_loss,
        hard_margin_loss,
        hard_margin_loss_type="soft_margin_debug",
        last_state_loss=False
        ):

    # compute the similarity score between text + image embedding and the text embedding (text + image) --> text
    
    # compute the similarity score between text + image embedding and the first frame embedding (text + image) --> text embedding
    batch_size = text_embedding.shape[0]

    if other_video_frame_embedding is not None:
        frame_embeddings = torch.cat([first_frame_embedding, last_frame_embedding, mid_frames_1_embedding, mid_frames_2_embedding, other_video_frame_embedding], dim=0)
        text_input_embedding = text_embedding.repeat(5, 1)
    else:
        frame_embeddings = torch.cat([first_frame_embedding, last_frame_embedding, mid_frames_1_embedding, mid_frames_2_embedding], dim=0)
        text_input_embedding = text_embedding.repeat(4, 1)
    import pdb ; pdb.set_trace()
    text_frame_embeddings = torch.cat([text_input_embedding, frame_embeddings], dim=1)
    text_frame_embeddings = transform_model(text_frame_embeddings)

    first_frame_embeddings = text_frame_embeddings[:batch_size]
    last_frame_embeddings = text_frame_embeddings[batch_size:2*batch_size]
    mid_frames_1_embeddings = text_frame_embeddings[2*batch_size:3*batch_size]
    mid_frames_2_embeddings = text_frame_embeddings[3*batch_size:4*batch_size]
    if other_video_frame_embedding is not None:
        other_video_frame_embeddings = text_frame_embeddings[4*batch_size:5*batch_size]

    first_frame_similarity = compute_similarity(text_embedding, first_frame_embeddings)
    last_frame_similarity = compute_similarity(text_embedding, last_frame_embeddings)
    mid_frames_1_similarity = compute_similarity(text_embedding, mid_frames_1_embeddings)
    mid_frames_2_similarity = compute_similarity(text_embedding, mid_frames_2_embeddings)

    if other_video_frame_embedding is not None:
        other_video_frame_similarity = compute_similarity(text_embedding, other_video_frame_embeddings)

    
    target = -1 * torch.ones(batch_size).to(first_frame_similarity.device)
    # sim first < sim middle 1 
    loss_first_mid1 = soft_margin_loss(first_frame_similarity, mid_frames_1_similarity, target)

    # sim first < sim middle 2 
    loss_first_mid2 = soft_margin_loss(first_frame_similarity, mid_frames_2_similarity, target)

    # sim first < sim last 
    loss_first_last = soft_margin_loss(first_frame_similarity, last_frame_similarity, target)

    # sim middle 1 < sim middle 2
    loss_mid1_mid2 = soft_margin_loss(mid_frames_1_similarity, mid_frames_2_similarity, target)

    # sim middle 1 < sim last
    loss_mid1_last = soft_margin_loss(mid_frames_1_similarity, last_frame_similarity, target)

    # sim middle 2 < sim last
    loss_mid2_last = soft_margin_loss(mid_frames_2_similarity, last_frame_similarity, target)

    if other_video_frame_embedding is not None:
        # sim other < sim first
        # loss_first_other = hard_margin_loss(other_video_frame_similarity, first_frame_similarity, target)
        if hard_margin_loss_type == "cross_entropy":
            cem_embeddings = torch.cat([first_frame_embeddings, other_video_frame_embeddings], dim=1)
            cem_target = torch.zeros(batch_size).to(first_frame_similarity.device)
            loss_first_other = hard_margin_loss(cem_embeddings, cem_target.long())
        elif hard_margin_loss_type == "soft_margin":
            loss_first_other = soft_margin_loss(other_video_frame_similarity, first_frame_similarity, target)
            # loss_last_other = soft_margin_loss(other_video_frame_similarity, last_frame_similarity, target)
            # loss_mid1_other = soft_margin_loss(other_video_frame_similarity, mid_frames_1_similarity, target)
            # loss_mid2_other = soft_margin_loss(other_video_frame_similarity, mid_frames_2_similarity, target)
            # other_loss = loss_first_other + loss_last_other + loss_mid1_other + loss_mid2_other

        elif hard_margin_loss_type == "hard_margin":
            loss_first_other = hard_margin_loss(other_video_frame_similarity, first_frame_similarity, target)
        else:
            raise ValueError(f"Loss type {hard_margin_loss_type} not supported")

    
    total_loss = loss_first_mid1 + loss_first_mid2 + loss_first_last + loss_mid1_mid2 + loss_mid1_last + loss_mid2_last
    if other_video_frame_embedding is not None:
        if hard_margin_loss_type == "cross_entropy":
            total_loss += loss_first_other * 0.1
        else:
            total_loss += loss_first_other * 10
            # total_loss += other_loss

    if last_state_loss:
        # reduce mse(last_frame_embedding, 1)
        l2_dis = F.mse_loss(last_frame_embedding, torch.ones_like(last_frame_embedding, requires_grad=True))
        total_loss += l2_dis * 0.01
        
        
        



    wandb_dict = {
        "rank_loss/loss_first_mid1": loss_first_mid1,
        "rank_loss/loss_first_mid2": loss_first_mid2,
        "rank_loss/loss_first_last": loss_first_last,
        "rank_loss/loss_mid1_mid2": loss_mid1_mid2,
        "rank_loss/loss_mid1_last": loss_mid1_last,
        "rank_loss/loss_mid2_last": loss_mid2_last,
        "similarity/first": torch.mean(first_frame_similarity).item(),
        "similarity/last": torch.mean(last_frame_similarity).item(),
        "similarity/mid1": torch.mean(mid_frames_1_similarity).item(),
        "similarity/mid2": torch.mean(mid_frames_2_similarity).item(),
    }
    if other_video_frame_embedding is not None:
        wandb_dict["rank_loss/loss_first_other"] = loss_first_other
        wandb_dict["similarity/other"] = torch.mean(other_video_frame_similarity).item()
    if last_state_loss:
        wandb_dict["rank_loss/last_state_loss"] = l2_dis

    return total_loss, wandb_dict




def DebugLoss(
        transform_model,
        text_embedding, 
        first_frame_embedding, 
        last_frame_embedding, 
        mid_frames_1_embedding, 
        mid_frames_2_embedding, 
        other_video_frame_embedding,
        soft_margin_loss,
        hard_margin_loss,
        hard_margin_loss_type="soft_margin_debug",
        last_state_loss=False
        ):

    # compute the similarity score between text + image embedding and the text embedding (text + image) --> text
    
    # compute the similarity score between text + image embedding and the first frame embedding (text + image) --> text embedding
    batch_size = text_embedding.shape[0]

    if other_video_frame_embedding is not None:
        frame_embeddings = torch.cat([first_frame_embedding, last_frame_embedding, mid_frames_1_embedding, mid_frames_2_embedding, other_video_frame_embedding], dim=0)
        text_input_embedding = text_embedding.repeat(5, 1)
    else:
        frame_embeddings = torch.cat([first_frame_embedding, last_frame_embedding, mid_frames_1_embedding, mid_frames_2_embedding], dim=0)
        text_input_embedding = text_embedding.repeat(4, 1)

    text_frame_embeddings = torch.cat([text_input_embedding, frame_embeddings], dim=1)
    text_frame_embeddings = transform_model(text_frame_embeddings)

    first_frame_embeddings = text_frame_embeddings[:batch_size]
    last_frame_embeddings = text_frame_embeddings[batch_size:2*batch_size]
    mid_frames_1_embeddings = text_frame_embeddings[2*batch_size:3*batch_size]
    mid_frames_2_embeddings = text_frame_embeddings[3*batch_size:4*batch_size]
    if other_video_frame_embedding is not None:
        other_video_frame_embeddings = text_frame_embeddings[4*batch_size:5*batch_size]

    first_frame_similarity = compute_similarity(text_embedding, first_frame_embeddings)
    last_frame_similarity = compute_similarity(text_embedding, last_frame_embeddings)
    mid_frames_1_similarity = compute_similarity(text_embedding, mid_frames_1_embeddings)
    mid_frames_2_similarity = compute_similarity(text_embedding, mid_frames_2_embeddings)

    if other_video_frame_embedding is not None:
        other_video_frame_similarity = compute_similarity(text_embedding, other_video_frame_embeddings)

    
    total_loss = 0
    
    target = -1 * torch.ones(batch_size).to(first_frame_similarity.device)
    # sim first < sim middle 1 
    loss_first_mid1 = soft_margin_loss(first_frame_similarity, mid_frames_1_similarity, target)

    # sim first < sim middle 2 
    loss_first_mid2 = soft_margin_loss(first_frame_similarity, mid_frames_2_similarity, target)

    # sim first < sim last 
    loss_first_last = soft_margin_loss(first_frame_similarity, last_frame_similarity, target)

    # sim middle 1 < sim middle 2
    loss_mid1_mid2 = soft_margin_loss(mid_frames_1_similarity, mid_frames_2_similarity, target)

    # sim middle 1 < sim last
    loss_mid1_last = soft_margin_loss(mid_frames_1_similarity, last_frame_similarity, target)

    # sim middle 2 < sim last
    loss_mid2_last = soft_margin_loss(mid_frames_2_similarity, last_frame_similarity, target)

    if other_video_frame_embedding is not None:
        # sim other < sim first
        # loss_first_other = hard_margin_loss(other_video_frame_similarity, first_frame_similarity, target)
        if hard_margin_loss_type == "cross_entropy":
            cem_embeddings = torch.cat([first_frame_similarity.unsqueeze(1), other_video_frame_similarity.unsqueeze(1)], dim=1)
            cem_target = torch.zeros(batch_size).to(first_frame_similarity.device)
            # cem_target = torch.ones(batch_size).to(first_frame_similarity.device)

            loss_first_other = hard_margin_loss(cem_embeddings, cem_target.long())
        elif hard_margin_loss_type == "soft_margin":
            loss_first_other = soft_margin_loss(other_video_frame_similarity, first_frame_similarity, target)
            # loss_last_other = soft_margin_loss(other_video_frame_similarity, last_frame_similarity, target)
            # loss_mid1_other = soft_margin_loss(other_video_frame_similarity, mid_frames_1_similarity, target)
            # loss_mid2_other = soft_margin_loss(other_video_frame_similarity, mid_frames_2_similarity, target)
            # other_loss = loss_first_other + loss_last_other + loss_mid1_other + loss_mid2_other

        elif hard_margin_loss_type == "hard_margin":
            loss_first_other = hard_margin_loss(other_video_frame_similarity, first_frame_similarity, target)
        else:
            raise ValueError(f"Loss type {hard_margin_loss_type} not supported")

    
    # total_loss = loss_first_mid1 + loss_first_mid2 + loss_first_last + loss_mid1_mid2 + loss_mid1_last + loss_mid2_last
    if other_video_frame_embedding is not None:
        if hard_margin_loss_type == "cross_entropy":
            total_loss += loss_first_other * 0.1
        else:
            total_loss += loss_first_other * 10
            # total_loss += other_loss

    if last_state_loss:
        # reduce mse(last_frame_embedding, 1)
        l2_dis = F.mse_loss(last_frame_similarity, torch.ones_like(last_frame_similarity))
        total_loss += l2_dis

    wandb_dict = {

        "similarity/first": torch.mean(first_frame_similarity).item(),
        "similarity/last": torch.mean(last_frame_similarity).item(),
        "similarity/mid1": torch.mean(mid_frames_1_similarity).item(),
        "similarity/mid2": torch.mean(mid_frames_2_similarity).item(),
        "similarity/other": torch.mean(other_video_frame_similarity).item(),

        "loss/total_loss": total_loss
    }
        
        
        



    # wandb_dict = {
    #     "rank_loss/loss_first_mid1": loss_first_mid1,
    #     "rank_loss/loss_first_mid2": loss_first_mid2,
    #     "rank_loss/loss_first_last": loss_first_last,
    #     "rank_loss/loss_mid1_mid2": loss_mid1_mid2,
    #     "rank_loss/loss_mid1_last": loss_mid1_last,
    #     "rank_loss/loss_mid2_last": loss_mid2_last,
    #     "similarity/first": torch.mean(first_frame_similarity).item(),
    #     "similarity/last": torch.mean(last_frame_similarity).item(),
    #     "similarity/mid1": torch.mean(mid_frames_1_similarity).item(),
    #     "similarity/mid2": torch.mean(mid_frames_2_similarity).item(),
    # }
    # if other_video_frame_embedding is not None:
    #     wandb_dict["rank_loss/loss_first_other"] = loss_first_other
    #     wandb_dict["similarity/other"] = torch.mean(other_video_frame_similarity).item()
    # if last_state_loss:
    #     wandb_dict["rank_loss/last_state_loss"] = l2_dis

    return total_loss, wandb_dict




    

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "Rank_loss_" + args.model_name + args.hard_margin_loss
    if args.sample_other_task:
        experiment_name += "_other_task"
    if args.last_state_loss:
        experiment_name += "_last_state_loss"

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Rank_loss_train_debug",
        config=args,
        name=experiment_name,
    )


    if args.model_name == "clip":
        transform_model = SingleLayerMLP(768 + 768, 768).to(device)
    elif args.model_name == "liv":
        transform_model = SingleLayerMLP(1024 + 1024, 1024).to(device)

    # if args.model_name == "clip":
    #     transform_model = ThreeLayerMLP(768 + 768, 768).to(device)
    # elif args.model_name == "liv":
    #     transform_model = ThreeLayerMLP(1024 + 1024, 1024).to(device)

    h5_file = h5py.File(args.h5_embedding_path, "r")
    dataset = ClipLivDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    soft_margin=1e-2
    hard_margin=1e-1

    soft_margin_loss = MarginRankingLoss(margin=soft_margin)
    if args.hard_margin_loss == "hard_margin" or args.hard_margin_loss == "soft_margin":
        hard_margin_loss = MarginRankingLoss(margin=hard_margin)
    elif args.hard_margin_loss == "cross_entropy":
        hard_margin_loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss type {args.hard_margin_loss} not supported")
    optimizer = torch.optim.Adam(transform_model.parameters(), lr=1e-3)

    other_video_frame = None
    for epoch in range(args.epochs):
        for data in tqdm(dataloader):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            first_frame = normalize_embeddings(data["first_frame"].to(device)).float()
            last_frame = normalize_embeddings(data["last_frame"].to(device)).float()
            mid_frames_1 = normalize_embeddings(data["mid_frames_1"].to(device)).float()
            mid_frames_2 = normalize_embeddings(data["mid_frames_2"].to(device)).float()
            if args.sample_other_task:
                other_video_frame = normalize_embeddings(data["other_video_frame"].to(device))


            loss, wandb_loss_dict = DebugLoss(
                transform_model,
                text_array,
                first_frame,
                last_frame,
                mid_frames_1,
                mid_frames_2,
                other_video_frame,
                soft_margin_loss,
                hard_margin_loss,
                hard_margin_loss_type=args.hard_margin_loss,
                last_state_loss=args.last_state_loss
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_loss_dict["loss/total_loss"] = loss
            wandb.log(wandb_loss_dict)








if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_25_for_clip_liv.h5')
    argparser.add_argument('--model_name', type=str, default='clip')
    argparser.add_argument('--sample_other_task', action='store_true')
    argparser.add_argument('--batch_size', type=int, default=10)
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--hard_margin_loss', type=str, choices=["soft_margin", "hard_margin", "cross_entropy"], default="soft_margin")
    argparser.add_argument('--last_state_loss', action='store_true')


    args = argparser.parse_args()
    main(args)




