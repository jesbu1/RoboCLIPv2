import torch
# from dataloader_liv_decoder_5_demo import video_collate_triangular_fn, LivVideoDecoderDataset5Frames
from dataset_clean import LivRealVideoTrainDataset, LivRealVideoEvalDataset
import torch.nn.functional as F
import numpy as np
import random
# from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import argparse
import wandb
from tqdm import tqdm
import h5py
from torch.nn import CrossEntropyLoss, BCELoss
import os
# from models_pe import ClassProgressTransformer as pe_model
from models_full_pe import ClassProgressTransformer as pe_model
# from models import ClassProgressTransformer as no_pe_model
# , RewardOneStepNewPositionEmbeddingPredictor
from utils_clean import update_model, CosineWithMinLRScheduler, eval_model
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math
from datetime import date
import pickle

from ignite.engine import Engine, Events
from ignite.handlers import EMAHandler, Checkpoint, DiskSaver
from train_roll_abrar import compute_metrics_multi, focal_loss, weighted_mse_loss, compute_metrics
from ema_utils import make_train_step_progress_fn
from ignite.contrib.handlers import ProgressBar
from torch.nn.functional import mse_loss
from eval_confusion_matrix_progress_only import plot_confusion_matrix
from eval_progress_progress_only import plot_progress



os.environ["TOKENIZERS_PARALLELISM"] = "False"



def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    experiment_name = str(args.extra_data_type) + "_full_PE"

    # experiment_name = str(args.extra_data_type) + "_DataType_" + str(args.data_type) + "_nolang_aug"
    # if args.full_set:
    #     experiment_name += "_FullSet"
    # if args.positional_encoding:
    #     experiment_name += "_PosEmb"
    # if args.last_frame_pe:
    #     experiment_name += "_LastFramePE"
    # experiment_name += "_View_" + args.view
    # if args.openx_data:
    #     experiment_name += "_OpenX"
    # else:
    #     experiment_name += "_NoOpenX"

    # # if args.extra_data_type == "metaworld":
    # #     experiment_name = "_NewPE_Crop_MetaWorld" 
    # # else: 
    # #     experiment_name = "_RealWorld_Koch"

    # experiment_name += "_Rewind_ratio_" + str(args.rewind_ratio)
    # experiment_name += "_EMA_momentum_" + str(args.ema_momentum)
    # if args.end_rewind_ratio > 0:
    #     experiment_name += "_End_Rewind_ratio_" + str(args.end_rewind_ratio)



    # if args.weighted_mse:
    #     experiment_name += "_weighted_mse"

    # experiment_name = "OneStep_" + experiment_name

    # if args.extra_data_type == "metaworld":
    #     group_name = "ProgressOnlyTest"
    # else:
    #     group_name = "RealWorld_Koch"
    # # get today date


    

    # group_name = "Dino_Koch_v2"
    group_name = args.extra_data_type + "_April_30"
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=group_name,
        config=args,
        name=experiment_name,
    )

    if args.extra_data_type == "metaworld":
        if args.text_embedding_model == "minilm":
            h5_train_eval_file = h5py.File("metaworld_dino_embeddings_train_5_demos.h5", "r")
            h5_eval_file = h5py.File("metaworld_dino_embeddings_eval_5_demos.h5", "r")
            extra_data_path = "metaworld_dino_embeddings_train_5_demos.h5"
    #    if args.text_embedding_model == "minilm":
    #         h5_train_eval_file = h5py.File("metaworld_dino_embeddings_train_5_demos_1_lang.h5", "r")
    #         h5_eval_file = h5py.File("metaworld_dino_embeddings_eval_5_demos.h5", "r")
    #         extra_data_path = "metaworld_dino_embeddings_train_5_demos_1_lang.h5"

    else:
        # h5_eval_file = h5py.File("jesse_collect_dataset_new_token.h5", "r")
        # extra_data_path = "jesse_collect_dataset_new_token.h5"
        if args.view == "side":
            h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_side_new_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_side_new_eval.h5", "r")
            if args.full_set:
                extra_data_path = "usc_koch_rewind_dino_reward_side_new.h5"
            else:
                extra_data_path = "usc_koch_rewind_dino_reward_side_new_train.h5"

        elif args.view == "top":
            h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_main_new_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_main_new_eval.h5", "r")
            if args.full_set:
                extra_data_path = "usc_koch_rewind_dino_reward_main_new.h5"
            else:
                extra_data_path = "usc_koch_rewind_dino_reward_main_new_train.h5"
        
        elif args.view == "all":
            h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_new_train_combine.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_new_eval_combine.h5", "r")
            if args.full_set:
                if args.data_type == "old":
                    extra_data_path = "usc_koch_rewind_dino_reward_all.h5"
                elif args.data_type == "new":
                    extra_data_path = "usc_koch_rewind_dino_reward_all_new.h5"
                elif args.data_type == "all":
                    extra_data_path = "usc_koch_rewind_dino_reward_all_combine.h5"  
                print("Using all data new generated")
            else:
                extra_data_path = "usc_koch_rewind_dino_reward_new_train_combine.h5"


        print("Using side view new generated", args.view)
        print("Using side view new generated", args.view)
        print("Using side view new generated", args.view)

    embedding_dim = 768

    if args.openx_data:
        openx_dataset = LivRealVideoTrainDataset(args, args.h5_embedding_path, split = False, sample_neg=False)
        if args.extra_data_type == "metaworld":
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        else:
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        
        openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
        extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

        openx_dataloader = DataLoader(openx_dataset, batch_size=openx_batch_size, shuffle=True, num_workers=int(args.worker * 8), drop_last=True, pin_memory=False)
        extra_dataloader = DataLoader(extra_dataset, batch_size=extra_batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=False)


        # h5_openx_eval_file = h5py.File("/home/jzhang96/full_openx_embeddings_v2_test.h5", "r")
        # # h5_openx_eval_file = h5py.File("/mnt/ssd_a_4tb/jzhang96/full_openx_embeddings_dino_test_backup.h5", "r")
        # # h5_openx_eval_file = h5py.File("/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_dino_test.h5", "r")

        # positive_eval_openx_dataset = LivRealVideoEvalDataset(args, 
        #                                                 h5_openx_eval_file,
        #                                                 label = "positive")
        # negative_eval_openx_dataset = LivRealVideoEvalDataset(args,
        #                                                 h5_openx_eval_file,
        #                                                 label = "negative")
        
        # openx_positive_eval_dataloader = DataLoader(positive_eval_openx_dataset, batch_size=args.batch_size // 8, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
        # openx_negative_eval_dataloader = DataLoader(negative_eval_openx_dataset, batch_size=args.batch_size // 8, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    else:
        # if args.extra_data_type == "metaworld":
        #     extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        # else:
        args.extra_data_ratio = 1
        extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)

        extra_dataloader = DataLoader(extra_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
        # positive_eval_openx_dataset = None
        # negative_eval_openx_dataset = None


    extra_eval_eval_pos_dataset = LivRealVideoEvalDataset(args, h5_eval_file, label = "positive", dataset = "extra")
    extra_eval_eval_neg_dataset = LivRealVideoEvalDataset(args, h5_eval_file, label = "negative", dataset = "extra")

    # confusion_matrix_name = "confusion_matrix_" + str(args.extra_data_type) + "_" + str(args.view) + ".h5"
    confusion_matrix_name = "test_h5_no_textaug1.h5"
    matrix_h5 = h5py.File(confusion_matrix_name, "w")


    video_dim = 768
    if args.text_embedding_model == "minilm":
        text_dim = 384
    elif args.text_embedding_model == "liv":
        text_dim = 1024
    else:
        raise ValueError("Invalid text embedding model")

    if args.positional_encoding:
        self_attention_model = pe_model(
            args=args,
            video_dim=video_dim,  # Original video embedding dimension
            text_dim=text_dim,   # Original text embedding dimension
            hidden_dim=512  # Common dimension for transformer processing
        ).to(device)
    else:
        self_attention_model = no_pe_model(
            args=args,
            video_dim=video_dim,  # Original video embedding dimension
            text_dim=text_dim,   # Original text embedding dimension
            hidden_dim=512  # Common dimension for transformer processing
        ).to(device)

    print(self_attention_model)
    if args.cosine_scheduler:
        base_optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineWithMinLRScheduler(base_optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)
    else:
        base_optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = None


    train_step_fn = make_train_step_progress_fn(
        self_attention_model=self_attention_model,
        optimizer=base_optimizer,
        scheduler=scheduler,
        args=args,
        device=device
    )
    print("Starting training")


    trainer = Engine(train_step_fn)
    # ema_handler = EMAHandler(self_attention_model, momentum=0.0002)
    ema_handler = EMAHandler(self_attention_model, momentum=args.ema_momentum)
    ema_model = ema_handler.ema_model
    ema_handler.attach(trainer, name="ema_momentum", event=Events.ITERATION_COMPLETED(every=1))
    
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    for epoch in range(args.epochs):

        self_attention_model.train()

        if args.openx_data:

            training_loader = zip(openx_dataloader, extra_dataloader)
            # call the ema trainer
            trainer.run(training_loader, max_epochs=1, epoch_length=len(openx_dataloader))

            ema_model.eval()
            self_attention_model.eval()
            with torch.no_grad():
                if epoch >= 17:
                    if args.extra_data_type == "metaworld":

                        plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, epoch = epoch, matrix_h5=matrix_h5, run_name = experiment_name)
                        plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, epoch = epoch, matrix_h5=matrix_h5, run_name = experiment_name)
                        plot_progress(h5_train_eval_file, "train", self_attention_model, args, epoch = epoch)
                        plot_progress(h5_eval_file, "eval", self_attention_model, args, epoch = epoch)

                        if epoch % 2 == 0:
                            compute_gif = True
                        else:
                            compute_gif = False

                        compute_metrics_multi(args, ema_model, threshold=0.5, compute_gif = compute_gif, epoch = epoch, one_step=True)

                    else: # real world data
                        # plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, epoch = epoch, ema=False)
                        # plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, epoch = epoch, ema=False)
                        plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = ema_model, args = args, epoch = epoch, ema=True, matrix_h5 = matrix_h5, run_name = experiment_name)
                        plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = ema_model, args = args, epoch = epoch, ema=True, matrix_h5 = matrix_h5, run_name = experiment_name)
                        plot_progress(h5_train_eval_file, "train", self_attention_model, args, epoch = epoch)
                        plot_progress(h5_eval_file, "eval", self_attention_model, args, epoch = epoch)

                        # save the model
                        model_dict = {
                            "model": self_attention_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "epoch": epoch,
                            "args": args
                        }
                        folder_name = "models/" + experiment_name
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        torch.save(model_dict, folder_name + "/model_" + str(epoch) + ".pth")


            ema_model.train()
            self_attention_model.train()

        else:
            training_loader = extra_dataloader
            # call the ema trainer
            trainer.run(training_loader, max_epochs=1, epoch_length=len(extra_dataloader))

            ema_model.eval()
            self_attention_model.eval()
            with torch.no_grad():
                if epoch >= 17:
                    if args.extra_data_type == "metaworld":

                        plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, epoch = epoch, ema = True, matrix_h5 = matrix_h5)
                        plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, epoch = epoch, ema = True, matrix_h5 = matrix_h5)
                        plot_progress(h5_train_eval_file, "train", self_attention_model, args, epoch = epoch)
                        plot_progress(h5_eval_file, "eval", self_attention_model, args, epoch = epoch)

                        if epoch % 2 == 0:
                            compute_gif = True
                        else:
                            compute_gif = False

                        compute_metrics_multi(args, ema_model, threshold=0.5, compute_gif = compute_gif, epoch = epoch, one_step=True)
                ema_model.train()
                self_attention_model.train()
        
            if args.extra_data_type == "metaworld":
                model_dict = {
                    "model": self_attention_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "epoch": epoch,
                    "args": args
                }
                folder_name = "models/" + experiment_name
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                torch.save(model_dict, folder_name + "/model_" + str(epoch) + ".pth")

                    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_droid_dino_train.h5')
    argparser.add_argument('--h5_embedding_path', type=str, default='/home/jzhang96/full_openx_embeddings_v2_train.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/mnt/ssd_a_4tb/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "real_world"], default="real_world")
    argparser.add_argument('--batch_size', type=int, default=1024)
    argparser.add_argument('--epochs', type=int, default=200)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=1)
    argparser.add_argument('--rewind', action='store_true')
    argparser.add_argument('--normalize_embedding', action='store_true')    
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--max_length', type=int, default=32)
    argparser.add_argument('--positional_encoding', action='store_true')
    argparser.add_argument('--openx_data', action='store_true')
    argparser.add_argument('--two_step_training', action='store_true')
    argparser.add_argument('--cosine_scheduler', action='store_true')
    argparser.add_argument('--clip_grad', action='store_true')
    argparser.add_argument('--progress_loss', action='store_true')
    argparser.add_argument('--view', type=str, default="side", choices=["side", "top", "all"])
    argparser.add_argument('--extra_data_ratio', type=float, default=0.02)
    argparser.add_argument('--catagorical_progress', action='store_true')

    argparser.add_argument('--text_embedding_model', type=str, default="minilm", choices=["minilm", "liv"])
    argparser.add_argument('--eval_interval', type=int, default=2)
    argparser.add_argument('--binary_threshold', type=float, default=0.5)
    argparser.add_argument('--rewind_ratio', type=float, default=0.5)
    argparser.add_argument('--progress_loss_weight', type=float, default=1)
    argparser.add_argument('--weighted_mse', action='store_true')
    argparser.add_argument('--last_frame_pe', action='store_true')
    argparser.add_argument('--ema_momentum', type=float, default=0.3)
    argparser.add_argument('--end_rewind_ratio', type=float, default=0.0)
    argparser.add_argument('--full_set', action='store_true')
    argparser.add_argument('--data_type', type=str, default="old", choices=["old", "new", "all"])



    args = argparser.parse_args()
    main(args)

