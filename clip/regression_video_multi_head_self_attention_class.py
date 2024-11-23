import torch
from PIL import Image
from dataloader_clipliv_video import video_class_collate_fn, ClipLivVideoClassDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from clip_utils import normalize_embeddings, compute_similarity, TwoLayerMLP, pca_learner, compute_M, SingleLayerMLP, ThreeLayerMLP
import wandb
from tqdm import tqdm
import h5py
from eval_video_self_attention_utils import plot_progress_class, plot_videos_class
from torch.nn import CrossEntropyLoss 
import os
from self_attention_utils import MultiHeadAttentionSubtractionClass

os.environ["TOKENIZERS_PARALLELISM"] = "False"





def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "CategoricalMultiHeadotherclass_" + args.model_name
    if args.sample_negative:
        experiment_name += "_sample_negative"



    experiment_name += "_heads_" + str(args.attention_heads)

    experiment_name += "_class_" + str(args.num_class)
    experiment_name += "_dropout_" + str(args.dropout)


    if args.pca:
        experiment_name += "_pca_" + str(args.pca_var)
        if args.pca_only_goal:
            experiment_name += "_only_goal"


    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="RegressionRandomStartMultiHeadSelfAttentioncompare",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")


    dataset = ClipLivVideoClassDataset(args, h5_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, collate_fn=video_class_collate_fn)

    loss_function = CrossEntropyLoss()


    # if args.model_name == "clip":
    #     transform_model = TwoLayerMLP(768).to(device)
    #     self_attention_model = MultiHeadAttention(768).to(device)

    # elif args.model_name == "liv":
    if args.sample_negative:
        num_class = args.num_class + 1
    else:
        num_class = args.num_class

    
        
    self_attention_model = MultiHeadAttentionSubtractionClass(1024, num_heads = args.attention_heads, class_num = num_class, dropout = args.dropout).to(device)

    optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)



    for epoch in range(args.epochs):
        self_attention_model.train()
        # total_loss = 0
        for i, data in enumerate(tqdm(dataloader)):
            text_array = normalize_embeddings(data["text_array"].to(device)).float()
            video_array = data["video_array"].to(device).float()
            
            # video
            mask = data["mask"].to(device).float()
            pred_class = self_attention_model(video_array, mask, text_array)
            gt_class = data["class_output"].to(device)

            accuracy = (torch.argmax(pred_class, dim = 1) == gt_class).float().mean()

            loss = loss_function(pred_class, gt_class)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log = {
                "loss": loss.item(),
                "accuracy": accuracy.item()
            }

            wandb.log(wandb_log)


        if epoch % 5 == 4:
            with torch.no_grad():
                self_attention_model.eval()
                # corr_train_dict = plot_progress_corr(h5_file, args.model_name, "train", self_attention_model)
                # corr_eval_dict = plot_progress_corr(h5_file, args.model_name, "eval", self_attention_model)
                # wandb.log(corr_train_dict)
                # wandb.log(corr_eval_dict)
                plot_progress_class(h5_file, args.model_name, "train", self_attention_model)
                plot_progress_class(h5_file, args.model_name, "eval", self_attention_model)
        #     # plot_wrong_progress(h5_file, args.model_name, "train", self_attention_model)
        #     # plot_wrong_progress(h5_file, args.model_name, "eval", self_attention_model)

        
        if epoch % 10 == 9:
            plot_videos_class(args.model_name, self_attention_model, num_class)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_25_for_clip_liv.h5')
    argparser.add_argument('--model_name', type=str, default='liv', choices=['clip', 'liv'])
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=100)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--pca_only_goal', action='store_true')
    argparser.add_argument('--pca_var', type=float, default=1.0)
    argparser.add_argument('--reverse', action='store_true')
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--sample_negative', action='store_true')
    argparser.add_argument('--num_class', type=int, default=10)
    args = argparser.parse_args()
    main(args)






    