import os
import h5py
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.train_utils import compute_metrics_multi, CosineWithMinLRScheduler
from model import ClassProgressTransformer 
from dataset import ReWiNDVideoDataset

from utils.eval_progress import plot_progress
from utils.ema_utils import make_train_step_progress_fn
from utils.eval_confusion_matrix import plot_confusion_matrix


from ignite.handlers import EMAHandler
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar




os.environ["TOKENIZERS_PARALLELISM"] = "False"



def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_dim = 768
    text_dim = 384

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    experiment_name = "ReWiND_Release" + str(args.extra_data_type)

    group_name = args.extra_data_type 
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=group_name,
        config=args,
        name=experiment_name,
    )

    if args.extra_data_type == "metaworld":
        train_file_name = os.path.join(args.h5_folder_path, "metaworld_train_embeddings.h5")
        eval_file_name = os.path.join(args.h5_folder_path, "metaworld_eval_embeddings.h5")
        h5_train_eval_file = h5py.File(train_file_name, "r")
        h5_eval_file = h5py.File(eval_file_name, "r")

    # else:
    #     h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_new_train_combine.h5", "r")
    #     h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_new_eval_combine.h5", "r")
    
    openx_h5_file = h5py.File(args.openx_embedding_path, "r")
    openx_dataset = ReWiNDVideoDataset(args, openx_h5_file, sample_neg=False)

    if args.extra_data_type == "metaworld":
        extra_dataset = ReWiNDVideoDataset(args, h5_train_eval_file, sample_neg=True)
    else:
        extra_dataset = ReWiNDVideoDataset(args, h5_train_eval_file, sample_neg=True)
    
    openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
    extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

    openx_dataloader = DataLoader(openx_dataset, batch_size=openx_batch_size, shuffle=True, num_workers=int(args.worker * 4), drop_last=True, pin_memory=False)
    extra_dataloader = DataLoader(extra_dataset, batch_size=extra_batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=False)

    self_attention_model = ClassProgressTransformer(
        args=args,
        video_dim=video_dim,  # Original video embedding dimension
        text_dim=text_dim,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)


    print(self_attention_model)
    base_optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineWithMinLRScheduler(base_optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)

    train_step_fn = make_train_step_progress_fn(
        self_attention_model=self_attention_model,
        optimizer=base_optimizer,
        scheduler=scheduler,
        args=args,
        device=device
    )
    print("Starting training")


    trainer = Engine(train_step_fn)
    ema_handler = EMAHandler(self_attention_model, momentum=args.ema_momentum)
    ema_model = ema_handler.ema_model
    ema_handler.attach(trainer, name="ema_momentum", event=Events.ITERATION_COMPLETED(every=1))
    
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    for epoch in range(args.epochs):

        self_attention_model.train()


        training_loader = zip(openx_dataloader, extra_dataloader)
        # call the ema trainer
        trainer.run(training_loader, max_epochs=1, epoch_length=len(openx_dataloader))

        ema_model.eval()
        self_attention_model.eval()
        with torch.no_grad():
            if args.extra_data_type == "metaworld":

                plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, epoch = epoch, run_name = experiment_name)
                plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, epoch = epoch, run_name = experiment_name)
                plot_progress(h5_train_eval_file, "train", self_attention_model, args, epoch = epoch)
                plot_progress(h5_eval_file, "eval", self_attention_model, args, epoch = epoch)



        ema_model.train()
        self_attention_model.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_folder_path', type=str, default='data/metaworld/')
    argparser.add_argument('--openx_embedding_path', type=str, default='/home/jzhang96/full_openx_embeddings_v2_train.h5', help="Path to the OpenX embeddings file")
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "real_world"], default="metaworld")
    argparser.add_argument('--batch_size', type=int, default=1024)
    argparser.add_argument('--epochs', type=int, default=200)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=1)
    argparser.add_argument('--rewind', action='store_true')
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--max_length', type=int, default=16)
    argparser.add_argument('--cosine_scheduler', action='store_true')
    argparser.add_argument('--clip_grad', action='store_true')
    argparser.add_argument('--extra_data_ratio', type=float, default=0.2)
    argparser.add_argument('--eval_interval', type=int, default=2)
    argparser.add_argument('--rewind_ratio', type=float, default=0.8)
    argparser.add_argument('--ema_momentum', type=float, default=0.3)

    args = argparser.parse_args()
    main(args)

