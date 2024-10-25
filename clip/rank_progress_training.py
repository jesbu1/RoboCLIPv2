import torch
from PIL import Image
from dataloader_clipliv import ClipLivDataset, ClipLivProgressDataset
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
def TripletLoss(
        transform_model,
        text_embedding, 
        goal_embedding, 
        progress_embedding, 
        progress, 
        margin_range = 0.5,
        ):

    # compute the similarity score between text + image embedding and the text embedding (text + image) --> text
    
    # compute the similarity score between text + image embedding and the first frame embedding (text + image) --> text embedding
    batch_size = text_embedding.shape[0]

    frame_embeddings = torch.cat([goal_embedding, progress_embedding], dim=0)
    text_input_embedding = text_embedding.repeat(2, 1)

    text_frame_embeddings = torch.cat([text_input_embedding, frame_embeddings], dim=1)
    text_frame_embeddings = transform_model(text_frame_embeddings)

    goal_embedding = text_frame_embeddings[:batch_size]
    progress_embedding = text_frame_embeddings[batch_size:2*batch_size]

    goal_similarity = compute_similarity(text_embedding, goal_embedding)
    progress_similarity = compute_similarity(text_embedding, progress_embedding)

    adaptive_margin = margin_range * progress

    loss = F.relu(-goal_similarity + progress_similarity + adaptive_margin).mean()



    wandb_dict = {
        "similarity/goal": torch.mean(goal_similarity).item(),
        "similarity/progress": torch.mean(progress_similarity).item(),
        "loss/goal_progress_loss": loss.item(),
    }


    return loss, wandb_dict






def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "TripletMarginLoss_" + args.model_name 


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
    dataset = ClipLivProgressDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(transform_model.parameters(), lr=1e-3)

    other_video_frame = None
    for epoch in range(args.epochs):
        for data in tqdm(dataloader):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            goal_array = normalize_embeddings(data["goal_array"].to(device)).float()
            progress_array = normalize_embeddings(data["progress_array"].to(device)).float()
            progress = data["progress"].to(device).float()


            loss, wandb_loss_dict = TripletLoss(
                transform_model,
                text_array,
                goal_array,
                progress_array,
                progress,
                margin_range=0.5
            )


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_loss_dict["loss/total_loss"] = loss
            wandb.log(wandb_loss_dict)

        if epoch % 50 == 0:
            plot_progress_eval(h5_file, args.model_name, transform_model)
            plot_progress_train(h5_file, args.model_name, transform_model)








if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_25_for_clip_liv.h5')
    argparser.add_argument('--model_name', type=str, default='clip')
    argparser.add_argument('--batch_size', type=int, default=10)
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--sample_other_task', action='store_true')
    argparser.add_argument('--hard_margin_loss', type=str, choices=["soft_margin", "hard_margin", "cross_entropy"], default="soft_margin")
    argparser.add_argument('--last_state_loss', action='store_true')

    args = argparser.parse_args()
    main(args)




