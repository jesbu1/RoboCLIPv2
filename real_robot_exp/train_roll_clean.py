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
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss, BCELoss
import os
from models import RewardTwoStepNewPositionEmbeddingPredictor
# , RewardOneStepNewPositionEmbeddingPredictor
from eval_confusion_matrix import plot_confusion_matrix
from eval_progress import plot_progress
from eval_raw_video_progress import real_video_plot
from utils_clean import update_model, CosineWithMinLRScheduler, eval_model
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import math


os.environ["TOKENIZERS_PARALLELISM"] = "False"







def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    if args.extra_data_type == "metaworld":
        experiment_name = "MetaWorld" 
    else: 
        experiment_name = "RealWorld_Koch"

    if args.openx_data:
        experiment_name += "_AddOpenXData"
    


    experiment_name += "_heads_" + str(args.attention_heads)


    if args.rewind:
        experiment_name += "_ReWind"
    if args.catagorical_progress:
        experiment_name += "_CatProgress"
    if args.subsample_video:
        experiment_name += "_SubVideo"
        experiment_name += "_MaxLen" + str(args.max_length)
    if args.positional_encoding:
        experiment_name += "_PosEmb"
    if args.two_step_training:
        experiment_name += "_TwoStep"
    else:
        experiment_name += "_OneStep"
    if args.layer_norm:
        experiment_name += "_LayerNorm"
    if args.first_frame_embedding:
        experiment_name += "_FirstFrameEmb"

    if args.learner_parameter:
        experiment_name += "_LearnerPara"
    if args.cosine_scheduler:
        experiment_name += "_CosScheduler"
    if args.clip_grad:
        experiment_name += "_ClipGrad"
    
    experiment_name += "_DecoderNum_" + str(args.decoder_num)
    experiment_name += "_epochs_" + str(args.epochs)
    experiment_name += "_lr_" + str(args.lr)
    # experiment_name += "_1_demo"
    
    
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="KochNoTokenModelV1",
        config=args,
        name=experiment_name,
    )



    # h5_file = h5py.File(args.h5_embedding_path, "r")
    if args.extra_data_type == "metaworld":
        h5_train_eval_file = h5py.File("metaworld_embedding_5_demo_dataset_v3_train.h5", "r")
        h5_eval_file = h5py.File("metaworld_embedding_5_demo_dataset_v3_eval.h5", "r")
        extra_data_path = "metaworld_embedding_5_demo_dataset_v3_train.h5"
    else:
        # h5_eval_file = h5py.File("jesse_collect_dataset_new_token.h5", "r")
        # extra_data_path = "jesse_collect_dataset_new_token.h5"
        h5_eval_file = h5py.File("usc_koch_rewind_reward.h5", "r")
        extra_data_path = "usc_koch_rewind_reward.h5"
    embedding_dim = 1024


    eval_dataset = None
    eval_dataloader = None
    if args.openx_data:
        openx_dataset = LivRealVideoTrainDataset(args, args.h5_embedding_path, split = False, sample_neg=False)
        if args.extra_data_type == "metaworld":
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        else:
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = True, sample_neg=True)
        openx_dataloader = DataLoader(openx_dataset, batch_size=args.batch_size * 3, shuffle=True, num_workers=int(args.worker * 2), drop_last=True, pin_memory=True)
        extra_dataloader = DataLoader(extra_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)

        positive_eval_openx_dataset = LivRealVideoEvalDataset(args, 
                                                        "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5",
                                                        label = "positive")
        negative_eval_openx_dataset = LivRealVideoEvalDataset(args,
                                                        "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5",
                                                        label = "negative")
        
        # positive_eval_openx_dataset = LivRealVideoEvalDataset(args, 
        #                                                 "/mnt/ssd_a_4tb/jzhang96/openx_embeddings_test_dataset_progrssed.h5",
        #                                                 label = "positive")
        # negative_eval_openx_dataset = LivRealVideoEvalDataset(args,
        #                                                 "/mnt/ssd_a_4tb/jzhang96/openx_embeddings_test_dataset_progrssed.h5",
        #                                                 label = "negative")
        
        openx_positive_eval_dataloader = DataLoader(positive_eval_openx_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
        openx_negative_eval_dataloader = DataLoader(negative_eval_openx_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    else:
        if args.extra_data_type == "metaworld":
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        else:
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = True, sample_neg=True)
        extra_dataloader = DataLoader(extra_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
        positive_eval_openx_dataset = None
        negative_eval_openx_dataset = None

        


    # elif args.extra_data:
    #     if args.extra_data_type == "metaworld":
    #         extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=args.demo_sample_neg)
    #     else:
    #         extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = True, sample_neg=args.demo_sample_neg)

    #     extra_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
    #     positive_eval_openx_dataset = None
    #     negative_eval_openx_dataset = None
    # elif args.openx_data:
    #     dataset = LivRealVideoTrainDataset(args, args.h5_embedding_path, split = False)
    #     openx_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
    #     positive_eval_openx_dataset = None
    #     negative_eval_openx_dataset = None
        
    # else:
    #     assert False, "No dataset specified"

    if args.extra_data_type == "metaworld":
        eval_file_name = "metaworld_embedding_5_demo_dataset_v3_eval.h5"
    else:
        # eval_file_name = "jesse_collect_dataset_new_token.h5"
        eval_file_name = "usc_koch_rewind_reward.h5"
    # extra_eval_train_pos_dataset = LivDemoVideoEvalDataset(args, extra_data_path, label = "positive", set_name="train")
    # extra_eval_train_neg_dataset = LivDemoVideoEvalDataset(args, extra_data_path, label = "negative", set_name="train")

    extra_eval_eval_pos_dataset = LivRealVideoEvalDataset(args, eval_file_name, label = "positive", dataset = "extra")
    extra_eval_eval_neg_dataset = LivRealVideoEvalDataset(args, eval_file_name, label = "negative", dataset = "extra")


    extra_eval_eval_pos_dataloader = DataLoader(extra_eval_eval_pos_dataset, batch_size=4, shuffle=True, num_workers=1, drop_last=False)
    extra_eval_eval_neg_dataloader = DataLoader(extra_eval_eval_neg_dataset, batch_size=4, shuffle=True, num_workers=1, drop_last=False)


    if args.two_step_training:
        classification_loss_function = BCELoss()
        if args.catagorical_progress:
            progress_loss_function = CrossEntropyLoss()
        else:
            progress_loss_function = mse_loss        
    elif args.catagorical_progress:
        progress_loss_function = CrossEntropyLoss()
        classification_loss_function = None
    else:
        progress_loss_function = mse_loss
        classification_loss_function = None



    if args.catagorical_progress :
        if not args.two_step_training:
            num_bins = args.catagorical_progress_bins + 1
        else:
            num_bins = args.catagorical_progress_bins
    else:
        num_bins = 1

    if args.two_step_training:
        self_attention_model = RewardTwoStepNewPositionEmbeddingPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
    else:
        self_attention_model = RewardOneStepNewPositionEmbeddingPredictor(embedding_dim, args = args, class_num=num_bins).to(device)


    print(self_attention_model)
    if args.cosine_scheduler:
        optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)
        scheduler = CosineWithMinLRScheduler(optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)
    else:
        optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)
        scheduler = None

    triangular_mask = torch.tril(torch.ones(args.max_length + 1, args.max_length + 1)).to(device).unsqueeze(0).unsqueeze(0)
    # else:
    #     triangular_mask = torch.tril(torch.ones(args.max_length, args.max_length)).to(device).unsqueeze(0).unsqueeze(0)

    
    for epoch in range(args.epochs):

        self_attention_model.train()

        if args.openx_data:
            for openx_data, extra_data in tqdm(zip(openx_dataloader, extra_dataloader), total = 100):
                '''
                data.keys:
                ['video_array', 'text_array', 'progress', 'class_label']
                video_array shape: torch.Size([batch_size, max_length, 1024])
                text_array shape: torch.Size([batch_size, 1024])
                progress shape: torch.Size([batch_size, max_length])
                class_label shape: torch.Size([batch_size, 1])

                '''
                optimizer.zero_grad()

                openx_len = len(openx_data["video_array"])
                extra_len = len(extra_data["video_array"])
                
                positive_video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
                positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"]], dim = 0).to(device).float()              
                positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)

                negative_video_array_1 = torch.roll(positive_video_array, args.batch_size, 0)
                negative_text_array_1 = positive_text_array.clone()
                negative_progress_1 = torch.zeros_like(positive_progress)                 

                openx_pos_video_array = torch.cat([positive_video_array[:openx_len], negative_video_array_1[:openx_len]], dim = 0)
                openx_pos_text_array = torch.cat([positive_text_array[:openx_len], negative_text_array_1[:openx_len]], dim = 0)
                openx_pos_progress = torch.cat([positive_progress[:openx_len], negative_progress_1[:openx_len]], dim = 0)
                    

                extra_pos_video_array = torch.cat([positive_video_array[openx_len:], negative_video_array_1[openx_len:]], dim = 0)
                extra_pos_text_array = torch.cat([positive_text_array[openx_len:], negative_text_array_1[openx_len:]], dim = 0)
                extra_pos_progress = torch.cat([positive_progress[openx_len:], negative_progress_1[openx_len:]], dim = 0)

                if args.two_step_training:
                    positive_class_label = torch.cat([openx_data["class_label"], extra_data["class_label"]], dim = 0).to(device)
                    negative_class_label_1 = torch.zeros_like(positive_class_label)
                    openx_pos_class_label = torch.cat([positive_class_label[:openx_len], negative_class_label_1[:openx_len]], dim = 0)
                    extra_pos_class_label = torch.cat([positive_class_label[openx_len:], negative_class_label_1[openx_len:]], dim = 0)
                    class_label = torch.cat([openx_pos_class_label, extra_pos_class_label], dim = 0)
                else:
                    class_label = None

                video_array = torch.cat([openx_pos_video_array, extra_pos_video_array], dim = 0)
                text_array = torch.cat([openx_pos_text_array, extra_pos_text_array], dim = 0)
                progress = torch.cat([openx_pos_progress, extra_pos_progress], dim = 0)
                

                openx_len = len(openx_pos_video_array)
                extra_len = len(extra_pos_video_array)
                batch_triangular_mask = triangular_mask.repeat(openx_len + extra_len, 1, 1, 1).bool()

                wandb_log, self_attention_model = update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label,
                classification_loss_function, progress_loss_function, optimizer, openx_len = openx_len, extra_len = extra_len, scheduler = scheduler)
                wandb.log(wandb_log)
                
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                wandb_log["lr"] = optimizer.param_groups[0]["lr"]

        else:
            for extra_data in tqdm(extra_dataloader):
                optimizer.zero_grad()
                video_array = extra_data["video_array"].to(device).float()
                text_array = extra_data["text_array"].to(device).float().squeeze(1)
                progress = extra_data["progress"].to(device)
                class_label = extra_data["class_label"].to(device)
                batch_triangular_mask = triangular_mask.repeat(video_array.size(0), 1, 1, 1).bool()
                wandb_log, self_attention_model = update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label,
                classification_loss_function, progress_loss_function, optimizer)
                wandb.log(wandb_log)

                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                wandb_log["lr"] = optimizer.param_groups[0]["lr"]

        # else:
        #     assert False, "No dataset specified"

        if epoch % 15 == 0:
            self_attention_model.eval()
            with torch.no_grad():
                if args.extra_data_type == "metaworld":
                    plot_progress(h5_train_eval_file, "train", self_attention_model, args)
                    plot_confusion_matrix(h5_file = h5_train_eval_file,
                                        set = "train",
                                        self_attention_model = self_attention_model,
                                        args = args)

                    plot_progress(h5_eval_file, "eval", self_attention_model, args)
                    plot_confusion_matrix(h5_file = h5_eval_file,
                                        set = "eval",
                                        self_attention_model = self_attention_model,
                                        args = args)                    

                else:
                    plot_progress(h5_eval_file, "train", self_attention_model, args)
                    plot_progress(h5_eval_file, "eval", self_attention_model, args)
                    plot_confusion_matrix(h5_file = h5_eval_file,
                                        set = "train",
                                        self_attention_model = self_attention_model,
                                        args = args)
                    plot_confusion_matrix(h5_file = h5_eval_file,
                                        set = "eval",
                                        self_attention_model = self_attention_model,
                                        args = args)
                    plot_confusion_matrix(h5_file = h5_eval_file, 
                                            set = "all", 
                                            self_attention_model = self_attention_model, 
                                            args = args)




                
                wandb_eval_log = {}

                if positive_eval_openx_dataset is not None:
                    class_accuracy, progress_loss = eval_model(openx_positive_eval_dataloader, self_attention_model, progress_loss_function, triangular_mask, args)
                    wandb_eval_log["openx_eval/progress_loss"] = progress_loss
                    if args.two_step_training:
                        wandb_eval_log["openx_eval/correct_class_accuracy"] = class_accuracy

                if negative_eval_openx_dataset is not None:
                    class_accuracy, progress_loss = eval_model(openx_negative_eval_dataloader, self_attention_model, progress_loss_function, triangular_mask, args)
                    wandb_eval_log["openx_eval/wrong_class_accuracy"] = class_accuracy


                # if extra_eval_train_pos_dataset is not None:
                #     class_accuracy, progress_loss = eval_model(extra_eval_train_pos_dataloader, self_attention_model, progress_loss_function, triangular_mask, args)
                #     wandb_eval_log["demo_dataset_train_set_correct_label/progress_loss"] = progress_loss
                #     wandb_eval_log["demo_dataset_train_set_correct_label/correct_class_accuracy"] = class_accuracy

        #         if extra_eval_train_neg_dataset is not None:
        #             class_accuracy, progress_loss = eval_model(extra_eval_train_neg_dataloader, self_attention_model, progress_loss_function, triangular_mask, args)
        #             wandb_eval_log["demo_dataset_train_set_wrong_label/wrong_class_accuracy"] = class_accuracy

                if extra_eval_eval_pos_dataset is not None:
                    class_accuracy, progress_loss = eval_model(extra_eval_eval_pos_dataloader, self_attention_model, progress_loss_function, triangular_mask, args)
                    wandb_eval_log["extra_eval/progress_loss"] = progress_loss
                    wandb_eval_log["extra_eval/correct_class_accuracy"] = class_accuracy

                if extra_eval_eval_neg_dataset is not None:
                    class_accuracy, progress_loss = eval_model(extra_eval_eval_neg_dataloader, self_attention_model, progress_loss_function, triangular_mask, args)
                    wandb_eval_log["extra_eval/wrong_class_accuracy"] = class_accuracy

                wandb.log(wandb_eval_log)




            


        # if epoch % 50 == 49:
        #     self_attention_model.eval()
        #     plot_videos_class("liv", self_attention_model, args)



            






if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/mnt/ssd_a_4tb/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "real_world"], default="real_world")
    argparser.add_argument('--batch_size', type=int, default=512)
    argparser.add_argument('--epochs', type=int, default=10000)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--rewind', action='store_true')
    argparser.add_argument('--normalize_embedding', action='store_true')    
    argparser.add_argument('--catagorical_progress', action='store_true')
    argparser.add_argument('--catagorical_progress_bins', type=int, default=5)
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--max_length', type=int, default=32)
    argparser.add_argument('--positional_encoding', action='store_true')
    argparser.add_argument('--openx_data', action='store_true')
    argparser.add_argument('--layer_norm', action='store_true')
    argparser.add_argument('--two_step_training', action='store_true')
    argparser.add_argument('--decoder_num', type=int, default=1)
    argparser.add_argument('--first_frame_embedding', action='store_true')
    argparser.add_argument('--learner_parameter', action='store_true')
    argparser.add_argument('--cosine_scheduler', action='store_true')
    argparser.add_argument('--clip_grad', action='store_true')
    argparser.add_argument('--progress_loss', action='store_true')

    args = argparser.parse_args()
    main(args)






    