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
from eval_rank_utils import plot_progress_eval, plot_progress_train

def RankCrossEntropyLoss(
        transform_model,
        text_embedding, 
        first_frame_embedding, 
        last_frame_embedding, 
        mid_frames_1_embedding, 
        mid_frames_2_embedding, 
        other_video_frame_embedding,
        loss_function,
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

    first_frame_similarity = compute_similarity(text_embedding, first_frame_embeddings).unsqueeze(1)
    last_frame_similarity = compute_similarity(text_embedding, last_frame_embeddings).unsqueeze(1)
    mid_frames_1_similarity = compute_similarity(text_embedding, mid_frames_1_embeddings).unsqueeze(1)
    mid_frames_2_similarity = compute_similarity(text_embedding, mid_frames_2_embeddings).unsqueeze(1)

    if other_video_frame_embedding is not None:
        other_video_frame_similarity = compute_similarity(text_embedding, other_video_frame_embeddings).unsqueeze(1)

    # compute the loss
    # cross entrpoy, sim last > sim mid 2 > sim mid 1 > sim first > sim other
    target = torch.zeros(batch_size).to(first_frame_similarity.device)

    emb_last = torch.cat([last_frame_similarity, mid_frames_2_similarity, mid_frames_1_similarity, first_frame_similarity], dim=1)
    if other_video_frame_embedding is not None:
        emb_last = torch.cat([emb_last, other_video_frame_similarity], dim=1)
    last_cross_emtropy_loss = loss_function(emb_last, target.long())

    # then for mid 2
    emb_mid2 = torch.cat([mid_frames_2_similarity, mid_frames_1_similarity, first_frame_similarity], dim=1)
    if other_video_frame_embedding is not None:
        emb_mid2 = torch.cat([emb_mid2, other_video_frame_similarity], dim=1)
    mid2_cross_entropy_loss = loss_function(emb_mid2, target.long())

    # then for mid 1
    emb_mid1 = torch.cat([mid_frames_1_similarity, first_frame_similarity], dim=1)
    if other_video_frame_embedding is not None:
        emb_mid1 = torch.cat([emb_mid1, other_video_frame_similarity], dim=1)
    mid1_cross_entropy_loss = loss_function(emb_mid1, target.long())

    # then for first
    if other_video_frame_embedding is not None:
        emb_first = torch.cat([first_frame_similarity, other_video_frame_similarity], dim=1)
        first_cross_entropy_loss = loss_function(emb_first, target.long())

    # mse_loss = torch.nn.MSELoss()
    # l2_dis = mse_loss(first_frame_similarity, last_frame_similarity)


    # total_loss = last_cross_emtropy_loss + mid2_cross_entropy_loss + mid1_cross_entropy_loss + l2_dis

    total_loss = last_cross_emtropy_loss + mid2_cross_entropy_loss + mid1_cross_entropy_loss
    if other_video_frame_embedding is not None:
        total_loss += first_cross_entropy_loss


    wandb_dict = {
        "loss/last_cross_entropy_loss": last_cross_emtropy_loss.item(),
        "loss/mid2_cross_entropy_loss": mid2_cross_entropy_loss.item(),
        "loss/mid1_cross_entropy_loss": mid1_cross_entropy_loss.item(),
        "loss/total_loss": total_loss.item(),
        "similarity/first": torch.mean(first_frame_similarity).item(),
        "similarity/last": torch.mean(last_frame_similarity).item(),
        "similarity/mid1": torch.mean(mid_frames_1_similarity).item(),
        "similarity/mid2": torch.mean(mid_frames_2_similarity).item(),
        # "loss/l2_dis": l2_dis.item(),
    }
    if other_video_frame_embedding is not None:
        wandb_dict["similarity/other"] = torch.mean(other_video_frame_similarity).item()
        wandb_dict["loss/first_cross_entropy_loss"] = first_cross_entropy_loss.item()
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
    experiment_name = "RankCrossEntropyLoss_" + args.model_name + args.hard_margin_loss + "add_mse"
    if args.sample_other_task:
        experiment_name += "_other_task"
    if args.last_state_loss:
        experiment_name += "_last_state_loss"

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Rank_loss_train",
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
    loss_function = torch.nn.CrossEntropyLoss()

    # soft_margin_loss = MarginRankingLoss(margin=soft_margin)
    # if args.hard_margin_loss == "hard_margin" or args.hard_margin_loss == "soft_margin":
    #     hard_margin_loss = MarginRankingLoss(margin=hard_margin)
    # elif args.hard_margin_loss == "cross_entropy":
    #     hard_margin_loss = torch.nn.CrossEntropyLoss()
    # else:
    #     raise ValueError(f"Loss type {args.hard_margin_loss} not supported")
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


            loss, wandb_loss_dict = RankCrossEntropyLoss(
                transform_model,
                text_array, 
                first_frame, 
                last_frame, 
                mid_frames_1, 
                mid_frames_2, 
                other_video_frame,
                loss_function,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_loss_dict["loss/total_loss"] = loss
            wandb.log(wandb_loss_dict)

        if epoch % 10 == 0:
            transform_model.eval()
            with torch.no_grad():
                plot_progress_eval(h5_file, args.model_name, transform_model)
                plot_progress_train(h5_file, args.model_name, transform_model)








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




