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
from models_pe import ClassProgressTransformer as pe_model
from models import ClassProgressTransformer as no_pe_model
# , RewardOneStepNewPositionEmbeddingPredictor
from eval_confusion_matrix_abrar import plot_confusion_matrix
from eval_progress_abrar import plot_progress
from utils_clean import update_model, CosineWithMinLRScheduler, eval_model
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math
from datetime import date
import pickle
from eval_rewind.eval_rewind_new import generate_rewind_data, generate_rewind_gif, compute_pearson_correlation_from_sequences
from eval_rewind.eval_rewind_new import plot_confusion_matrix_from_predictions, compute_mse_from_sequences, compute_spearman_correlation_from_sequences
from eval_rewind.eval_rewind_new import compute_spearman_correlation_multi_annotations, rank_comparison

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce_loss = F.binary_cross_entropy(pred.squeeze(), target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1-pt)**gamma * bce_loss
    return focal_loss.mean()


def weighted_mse_loss(pred, target, weight_factor=2.0):
    # weight based on target value (uncomment)
    # weights = 1.0 + weight_factor * target  # Higher targets get higher weights
    
    # weight based on position in sequence (uncomment)
    weights = torch.linspace(1, weight_factor, target.shape[1]).unsqueeze(0).expand_as(target).to(target.device)

    squared_diff = (pred - target) ** 2
    weighted_squared_diff = weights * squared_diff
    return weighted_squared_diff.mean()


def compute_metrics(predictions, targets):
    """Compute classification metrics"""
    # Convert predictions to binary (0 or 1)
    binary_preds = (predictions >= 0.5).float()
    
    # Convert to numpy for sklearn metrics
    binary_preds = binary_preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(targets, binary_preds)
    precision = precision_score(targets, binary_preds)
    recall = recall_score(targets, binary_preds)
    f1 = f1_score(targets, binary_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_metrics_multi(args, self_attention_model, threshold, compute_gif = False, epoch = None):

    for file in os.listdir("./"):
        if file.endswith(".pkl"):
            os.remove(file)
    confusion_matrix, all_seqs, tasks, text_list = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end.pkl",
        args = args,
        one_step = False,
        threshold = threshold
    )
    os.remove("final_rewind_cache_oxe_pos_end.pkl")

    confusion_matrix_1, all_seqs1, _, _ = generate_rewind_data(
            h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
            json_path="new_task_v2.json",
            set_type="eval",
            rewind_model=self_attention_model,
            cache_path="final_rewind_cache_oxe_pos_end_1.pkl",
            args = args,
            annotation = 1,
            one_step = False,
            threshold = threshold
        )
    os.remove("final_rewind_cache_oxe_pos_end_1.pkl")

    confusion_matrix_2, all_seqs2, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_2.pkl",
        args = args,
        annotation = 2,
        one_step = False,
        threshold = threshold
    )
    os.remove("final_rewind_cache_oxe_pos_end_2.pkl")

    confusion_matrix_3, all_seqs3, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_3.pkl",
        args = args,
        annotation = 3,
        one_step = False,
        threshold = threshold
    )
    os.remove("final_rewind_cache_oxe_pos_end_3.pkl")

    confusion_matrix_all_fail, _, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval_fail.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_fail.pkl",
        args = args,
        one_step = False,
        threshold = threshold
    )
    os.remove("final_rewind_cache_oxe_pos_end_fail.pkl")

    confusion_matrix_close_success, _, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval_close_succ.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_close_succ.pkl",
        args = args,
        one_step = False,
        threshold = threshold
    )
    os.remove("final_rewind_cache_oxe_pos_end_close_succ.pkl")


    compute_pearson_correlation_from_sequences(
        all_seqs=all_seqs,
        set_type="eval",
        project_name="roboclip-v2",
        env_names=tasks,
        threshold=threshold,
        epoch=epoch
    )

    plot_confusion_matrix_from_predictions(
        predicted_rewards=confusion_matrix,
        task_names=tasks,
        set_type="eval",
        text_instructions=text_list,
        fig_name="Rewind",
        threshold=threshold,
        epoch=epoch
    )


    # # ============ 4) 计算 MSE ============
    compute_mse_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval",
        threshold=threshold,
        epoch=epoch
    )

    # ============ 5) 计算 Spearman 相关系数 ============
    compute_spearman_correlation_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval",
        threshold=threshold,
        epoch=epoch
    )

    # compute_spearman_correlation_from_sequences(
    #     all_seqs=all_seqs1,
    #     env_names=tasks,
    #     set_type="eval",
    #     threshold=threshold,
    #     epoch=epoch
    # )

    # compute_spearman_correlation_from_sequences(
    #     all_seqs=all_seqs2,
    #     env_names=tasks,
    #     set_type="eval",
    #     threshold=threshold,
    #     epoch=epoch
    # )

    # compute_spearman_correlation_from_sequences(
    #     all_seqs=all_seqs3,
    #     env_names=tasks,
    #     set_type="eval",
    #     threshold=threshold,
    #     epoch=epoch
    # )

    compute_spearman_correlation_multi_annotations(
        all_seqs_a=all_seqs1,
        all_seqs_b=all_seqs2,
        all_seqs_c=all_seqs3,
        all_seqs_d=all_seqs,
        env_names=tasks,
        set_type="eval",
        threshold=threshold,
        epoch=epoch
    )

    rank_comparison(confusion_matrix_all_fail, confusion_matrix_close_success, confusion_matrix, threshold, epoch=epoch)


    if compute_gif:
        generate_rewind_gif(
            h5_path="eval_rewind/metaworld_dino_embeddings_eval_close_succ_128.h5",
            json_path="new_task_v2.json",
            set_type="eval",
            rewind_model=self_attention_model,
            device="cuda",
            args=args,
            threshold=threshold,
            epoch=epoch
        )




def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    if args.extra_data_type == "metaworld":
        experiment_name = "NewPE_Crop_MetaWorld" 
    else: 
        experiment_name = "RealWorld_Koch"

    experiment_name += "_binary_thrd_" + str(args.binary_threshold)
    experiment_name += "_Rewind_ratio_" + str(args.rewind_ratio)



    if args.text_embedding_model == "minilm":
        experiment_name += "_MiniLM"
    elif args.text_embedding_model == "liv":
        experiment_name += "_Liv"



    if args.openx_data:
        experiment_name += "_AddOpenXData"

    if args.rewind:
        experiment_name += "_ReWind"
    if args.subsample_video:
        experiment_name += "_SubVideo"
        experiment_name += "_MaxLen" + str(args.max_length)
    if args.positional_encoding:
        experiment_name += "_PosEmb"
    if args.last_frame_pe:
        experiment_name += "_LastFramePE"

    experiment_name += "_View_" + str(args.view)
    experiment_name += "_ExtraDataRatio_" + str(args.extra_data_ratio)
    
    experiment_name += "_epochs_" + str(args.epochs)
    experiment_name += "_lr_" + str(args.lr)
    experiment_name += "_progress_loss_weight_" + str(args.progress_loss_weight)
    if args.weighted_mse:
        experiment_name += "_weighted_mse"

    
    if args.extra_data_type == "metaworld":
        group_name = "FixRewindnewlog_2step_Crop_MetaWorldNew"
    else:
        group_name = "RealWorld_Koch"
    # get today date


    

    # group_name = "Dino_Koch_v2"
    group_name = args.extra_data_type + "_NewAblate_" + group_name
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=group_name,
        config=args,
        name=experiment_name,
    )

    if args.extra_data_type == "metaworld":
        h5_train_eval_file = h5py.File("metaworld_dino_embeddings_train.h5", "r")
        h5_eval_file = h5py.File("metaworld_dino_embeddings_eval.h5", "r")
        extra_data_path = "metaworld_dino_embeddings_train.h5"
    else:
        # h5_eval_file = h5py.File("jesse_collect_dataset_new_token.h5", "r")
        # extra_data_path = "jesse_collect_dataset_new_token.h5"
        if args.view == "side":
            h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_side_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_side_eval.h5", "r")
            extra_data_path = "usc_koch_rewind_dino_reward_side_train.h5"
        elif args.view == "top":
            h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_main_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_main_eval.h5", "r")
            extra_data_path = "usc_koch_rewind_dino_reward_main_train.h5"
    embedding_dim = 768

    if args.openx_data:
        openx_dataset = LivRealVideoTrainDataset(args, args.h5_embedding_path, split = False, sample_neg=False)
        if args.extra_data_type == "metaworld":
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        else:
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        
        openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
        extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

        openx_dataloader = DataLoader(openx_dataset, batch_size=openx_batch_size, shuffle=True, num_workers=int(args.worker * 18), drop_last=True, pin_memory=True)
        extra_dataloader = DataLoader(extra_dataset, batch_size=extra_batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)


        h5_openx_eval_file = h5py.File("/home/jzhang96/full_openx_embeddings_v2_test.h5", "r")
        # h5_openx_eval_file = h5py.File("/mnt/ssd_a_4tb/jzhang96/full_openx_embeddings_dino_test_backup.h5", "r")
        # h5_openx_eval_file = h5py.File("/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_dino_test.h5", "r")

        positive_eval_openx_dataset = LivRealVideoEvalDataset(args, 
                                                        h5_openx_eval_file,
                                                        label = "positive")
        negative_eval_openx_dataset = LivRealVideoEvalDataset(args,
                                                        h5_openx_eval_file,
                                                        label = "negative")
        
        openx_positive_eval_dataloader = DataLoader(positive_eval_openx_dataset, batch_size=args.batch_size // 8, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
        openx_negative_eval_dataloader = DataLoader(negative_eval_openx_dataset, batch_size=args.batch_size // 8, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    else:
        # if args.extra_data_type == "metaworld":
        #     extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        # else:
        extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)

        extra_dataloader = DataLoader(extra_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
        positive_eval_openx_dataset = None
        negative_eval_openx_dataset = None
        openx_positive_eval_dataloader = None
        openx_negative_eval_dataloader = None




    extra_eval_eval_pos_dataset = LivRealVideoEvalDataset(args, h5_eval_file, label = "positive", dataset = "extra")
    extra_eval_eval_neg_dataset = LivRealVideoEvalDataset(args, h5_eval_file, label = "negative", dataset = "extra")

    if args.openx_data:
        extra_eval_eval_pos_dataloader = DataLoader(extra_eval_eval_pos_dataset, batch_size=5, shuffle=True, num_workers=1, drop_last=True)
        extra_eval_eval_neg_dataloader = DataLoader(extra_eval_eval_neg_dataset, batch_size=5, shuffle=True, num_workers=1, drop_last=True)
    else:
        extra_eval_eval_pos_dataloader = DataLoader(extra_eval_eval_pos_dataset, batch_size=5, shuffle=True, num_workers=0, drop_last=True)
        extra_eval_eval_neg_dataloader = DataLoader(extra_eval_eval_neg_dataset, batch_size=5, shuffle=True, num_workers=0, drop_last=True)


    # progress_loss_function = mse_loss
    if args.weighted_mse:
        progress_loss_function = weighted_mse_loss
    else:
        progress_loss_function = mse_loss

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
                base_optimizer.zero_grad()

                openx_len = len(openx_data["video_array"])
                extra_len = len(extra_data["video_array"])

                positive_video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
                positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"].squeeze()], dim = 0).to(device).float()              
                positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)
                positive_progress_mask = torch.ones_like(positive_progress).bool()


                negative_video_array_1 = torch.roll(positive_video_array, extra_len, 0)
                negative_text_array_1 = positive_text_array.clone()
                negative_progress_1 = torch.zeros_like(positive_progress)
                negative_progress_mask_1 = torch.ones_like(negative_progress_1).bool()
                # stop_idx = round(negative_progress_1.size(1) * args.negative_mask_ratio)
                # negative_progress_mask_1[:, :stop_idx] = False


                openx_pos_video_array = torch.cat([positive_video_array[:openx_len], negative_video_array_1[:openx_len]], dim = 0)
                openx_pos_text_array = torch.cat([positive_text_array[:openx_len], negative_text_array_1[:openx_len]], dim = 0)
                openx_pos_progress = torch.cat([positive_progress[:openx_len], negative_progress_1[:openx_len]], dim = 0)
                openx_pos_progress_mask = torch.cat([positive_progress_mask[:openx_len], negative_progress_mask_1[:openx_len]], dim = 0)
                    

                extra_pos_video_array = torch.cat([positive_video_array[openx_len:], negative_video_array_1[openx_len:]], dim = 0)
                extra_pos_text_array = torch.cat([positive_text_array[openx_len:], negative_text_array_1[openx_len:]], dim = 0)
                extra_pos_progress = torch.cat([positive_progress[openx_len:], negative_progress_1[openx_len:]], dim = 0)
                extra_pos_progress_mask = torch.cat([positive_progress_mask[openx_len:], negative_progress_mask_1[openx_len:]], dim = 0)


                video_array = torch.cat([openx_pos_video_array, extra_pos_video_array], dim = 0)
                text_array = torch.cat([openx_pos_text_array, extra_pos_text_array], dim = 0)
                progress = torch.cat([openx_pos_progress, extra_pos_progress], dim = 0).float()

                openx_len = len(openx_pos_video_array)
                extra_len = len(extra_pos_video_array)

                video_embedding = video_array

                # Binary classification targets
                compressed_extra_class_label = extra_data["class_label"][:, 0].float()
                openx_target = torch.cat([torch.ones(openx_len // 2), torch.zeros(openx_len // 2)], dim=0).to(device)
                extra_target = torch.cat([compressed_extra_class_label, torch.zeros(extra_len // 2)], dim=0).to(device)


                # Get predictions from classifier
                progress_pred, class_pred = self_attention_model(video_embedding, text_array)
                openx_pred = class_pred[:openx_len]
                extra_pred = class_pred[openx_len:]
                
                # Calculate focal loss to handle class imbalance

                if args.two_step_training:
                    openx_loss = focal_loss(openx_pred, openx_target)
                    extra_loss = focal_loss(extra_pred, extra_target)


                openx_progress_pred = progress_pred[:openx_len]
                extra_progress_pred = progress_pred[openx_len:]

                openx_progress_target = progress[:openx_len]
                extra_progress_target = progress[openx_len:]
                
                valid_openx_progress_pred = openx_progress_pred[openx_target.bool()]
                valid_openx_progress_target = openx_progress_target[openx_target.bool()]

                valid_extra_progress_pred = extra_progress_pred[extra_target.bool()]
                valid_extra_progress_target = extra_progress_target[extra_target.bool()]


                # Add progress prediction loss if applicable
                if args.catagorical_progress:
                    assert "not supported yet"
                else:
                    openx_progress_loss = progress_loss_function(valid_openx_progress_pred[:,1:].squeeze(), valid_openx_progress_target[:,1:])
                    extra_progress_loss = progress_loss_function(valid_extra_progress_pred[:,1:].squeeze(), valid_extra_progress_target[:,1:])

                openx_ratio = len(valid_openx_progress_pred) / (len(valid_openx_progress_pred) + len(valid_extra_progress_pred))
                extra_ratio = len(valid_extra_progress_pred) / (len(valid_openx_progress_pred) + len(valid_extra_progress_pred))
                progress_loss = openx_progress_loss * openx_ratio + extra_progress_loss * extra_ratio
                loss = openx_loss * openx_ratio + extra_loss * extra_ratio + progress_loss * args.progress_loss_weight

                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), 1.0)
                base_optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                # Log all metrics

                wandb_log = {
                    "train/openx_class_loss": openx_loss.item(),
                    "train/extra_class_loss": extra_loss.item(),
                    "train/progress_loss": progress_loss.item(),
                    "train/openx_progress_loss": openx_progress_loss.item(),
                    "train/extra_progress_loss": extra_progress_loss.item(),
                    "train/total_loss": loss.item(),
                    "lr": base_optimizer.param_groups[0]["lr"],
                    # OpenX metrics
                    "train/openx_accuracy": compute_metrics(openx_pred.squeeze(), openx_target)['accuracy'],
                    "train/openx_precision": compute_metrics(openx_pred.squeeze(), openx_target)['precision'],
                    "train/openx_recall": compute_metrics(openx_pred.squeeze(), openx_target)['recall'],
                    "train/openx_f1": compute_metrics(openx_pred.squeeze(), openx_target)['f1'],
                    # Extra metrics
                    "train/extra_accuracy": compute_metrics(extra_pred.squeeze(), extra_target)['accuracy'],
                    "train/extra_precision": compute_metrics(extra_pred.squeeze(), extra_target)['precision'],
                    "train/extra_recall": compute_metrics(extra_pred.squeeze(), extra_target)['recall'],
                    "train/extra_f1": compute_metrics(extra_pred.squeeze(), extra_target)['f1'],
                    # # Combined metrics (weighted average)
                    "train/combined_accuracy": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['accuracy'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['accuracy'],
                    "train/combined_precision": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['precision'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['precision'],
                    "train/combined_recall": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['recall'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['recall'],
                    "train/combined_f1": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['f1'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['f1']
                }
                wandb.log(wandb_log)
            
        else:
            for extra_data in tqdm(extra_dataloader):
                '''
                data.keys:
                ['video_array', 'text_array', 'progress', 'class_label']
                video_array shape: torch.Size([batch_size, max_length, 1024])
                text_array shape: torch.Size([batch_size, 1024])
                progress shape: torch.Size([batch_size, max_length])
                class_label shape: torch.Size([batch_size, 1])

                '''
                base_optimizer.zero_grad()

                extra_len = len(extra_data["video_array"])

                video_array = extra_data["video_array"].to(device).float()
                text_array = extra_data["text_array"].squeeze(1).to(device).float()
                progress = extra_data["progress"].to(device).float()
                progress_mask = torch.ones_like(progress).bool()




                video_embedding = video_array

                # Binary classification targets
                compressed_extra_class_label = extra_data["class_label"][:, 0].float()
                extra_target = compressed_extra_class_label.to(device)

                # Get predictions from classifier
                progress_pred, class_pred = self_attention_model(video_embedding, text_array)
                extra_pred = class_pred

                # Calculate focal loss to handle class imbalance
                extra_loss = focal_loss(extra_pred, extra_target)

                extra_progress_pred = progress_pred
                extra_progress_target = progress

                valid_extra_progress_pred = extra_progress_pred[extra_target.bool()]
                valid_extra_progress_target = extra_progress_target[extra_target.bool()]

                # Add progress prediction loss if applicable
                if args.catagorical_progress:
                    assert "not supported yet"
                else:
                    extra_progress_loss = progress_loss_function(valid_extra_progress_pred[:,1:].squeeze(), valid_extra_progress_target[:,1:])

                loss = extra_loss + extra_progress_loss * args.progress_loss_weight

                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), 1.0)
                base_optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                # Log all metrics

                wandb_log = {
                    "train/extra_class_loss": extra_loss.item(),
                    "train/progress_loss": extra_progress_loss.item(),
                    "train/total_loss": loss.item(),
                    "lr": base_optimizer.param_groups[0]["lr"],
                    # Extra metrics
                    "train/extra_accuracy": compute_metrics(extra_pred.squeeze(), extra_target)['accuracy'],
                    "train/extra_precision": compute_metrics(extra_pred.squeeze(), extra_target)['precision'],
                    "train/extra_recall": compute_metrics(extra_pred.squeeze(), extra_target)['recall'],
                    "train/extra_f1": compute_metrics(extra_pred.squeeze(), extra_target)['f1'],
                }



        # Evaluation
        if epoch % args.eval_interval == 0:  # Only evaluate at specified intervals

            print(f"\nRunning evaluation at epoch {epoch}")
            with torch.no_grad():

                self_attention_model.eval()
                wandb_eval_log = {}
                
                # OpenX Evaluation
                print("\nEvaluating OpenX dataset:")
                if openx_positive_eval_dataloader is not None:
                    print("- Evaluating OpenX positive samples")
                if openx_negative_eval_dataloader is not None:
                    print("- Evaluating OpenX negative samples")
                
                # Initialize evaluation lists
                openx_positive_eval_losses = []
                openx_positive_eval_preds = []
                openx_positive_eval_targets = []
                openx_positive_eval_progress_losses = []
                
                openx_negative_eval_losses = []
                openx_negative_eval_preds = []
                openx_negative_eval_targets = []
                openx_negative_eval_progress_losses = []

                
                extra_eval_eval_pos_losses = []
                extra_eval_eval_pos_preds = []
                extra_eval_eval_pos_targets = []
                extra_eval_eval_pos_progress_losses = []
                extra_eval_eval_pos_progress_preds = []
                extra_eval_eval_pos_progress_targets = []
                
                extra_eval_eval_neg_losses = []
                extra_eval_eval_neg_preds = []
                extra_eval_eval_neg_targets = []
                extra_eval_eval_neg_progress_losses = []

                # Evaluate OpenX positive samples
                if openx_positive_eval_dataloader is not None:
                    for data in openx_positive_eval_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions

                        progress_pred, class_pred = self_attention_model(video_array, text_array)
                        target = torch.ones(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        openx_positive_eval_losses.append(loss.item())
                        openx_positive_eval_preds.extend(class_pred.squeeze().cpu().numpy())
                        openx_positive_eval_targets.extend(target.cpu().numpy())
                        
                        # Progress loss
                        if args.catagorical_progress:
                            assert "not supported yet"
                        else:
                            progress_loss = progress_loss_function(progress_pred[:,1:].squeeze(), progress_target[:,1:])
                        
                        openx_positive_eval_progress_losses.append(progress_loss.item())
                        

                # Evaluate OpenX negative samples
                if openx_negative_eval_dataloader is not None:
                    for data in openx_negative_eval_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions

                        progress_pred, class_pred = self_attention_model(video_array, text_array)
                        target = torch.zeros(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        openx_negative_eval_losses.append(loss.item())
                        openx_negative_eval_preds.extend(class_pred.squeeze().cpu().numpy())
                        openx_negative_eval_targets.extend(target.cpu().numpy())
                        
        #                 # Progress loss
        #                 if args.catagorical_progress:
        #                     assert "not supported yet"
        #                 else:
        #                     progress_loss = progress_loss_function(progress_pred[:,1:].squeeze(), progress_target[:,1:])
                        
        #                 openx_negative_eval_progress_losses.append(progress_loss.item())

                # Log metrics only if we have data
                if len(openx_positive_eval_preds) > 0 or len(openx_negative_eval_preds) > 0:
                    # OpenX Combined Classification Metrics
                    openx_preds = np.array(openx_positive_eval_preds + openx_negative_eval_preds)
                    openx_targets = np.array(openx_positive_eval_targets + openx_negative_eval_targets)
                    
                    wandb_eval_log["openx_eval/loss"] = (np.mean(openx_positive_eval_losses) if len(openx_positive_eval_losses) > 0 else 0) + \
                                                      (np.mean(openx_negative_eval_losses) if len(openx_negative_eval_losses) > 0 else 0) / 2
                    
                    if len(openx_preds) > 0:
                        wandb_eval_log["openx_eval/f1"] = f1_score(openx_targets > args.binary_threshold, openx_preds > args.binary_threshold)
                        wandb_eval_log["openx_eval/precision"] = precision_score(openx_targets > args.binary_threshold, openx_preds > args.binary_threshold)
                        wandb_eval_log["openx_eval/recall"] = recall_score(openx_targets > args.binary_threshold, openx_preds > args.binary_threshold)
                        wandb_eval_log["openx_eval/accuracy"] = accuracy_score(openx_targets > args.binary_threshold, openx_preds > args.binary_threshold)
                    
                    # OpenX Positive Sample Metrics
                    if len(openx_positive_eval_preds) > 0:
                        wandb_eval_log["openx_eval/positive_loss"] = np.mean(openx_positive_eval_losses)
                        wandb_eval_log["openx_eval/positive_f1"] = f1_score(np.ones_like(openx_positive_eval_targets), 
                                                                          np.array(openx_positive_eval_preds) > args.binary_threshold)
                        wandb_eval_log["openx_eval/positive_precision"] = precision_score(np.ones_like(openx_positive_eval_targets), 
                                                                                       np.array(openx_positive_eval_preds) > args.binary_threshold)
                        wandb_eval_log["openx_eval/positive_recall"] = recall_score(np.ones_like(openx_positive_eval_targets), 
                                                                                 np.array(openx_positive_eval_preds) > args.binary_threshold)
                        wandb_eval_log["openx_eval/positive_accuracy"] = accuracy_score(np.ones_like(openx_positive_eval_targets), 
                                                                                    np.array(openx_positive_eval_preds) > args.binary_threshold)
                    
                    # OpenX Negative Sample Metrics
                    if len(openx_negative_eval_preds) > 0:
                        wandb_eval_log["openx_eval/negative_loss"] = np.mean(openx_negative_eval_losses)
                        wandb_eval_log["openx_eval/negative_f1"] = f1_score(np.zeros_like(openx_negative_eval_targets), 
                                                                          np.array(openx_negative_eval_preds) > args.binary_threshold)
                        wandb_eval_log["openx_eval/negative_precision"] = precision_score(np.zeros_like(openx_negative_eval_targets), 
                                                                                       np.array(openx_negative_eval_preds) > args.binary_threshold)
                        wandb_eval_log["openx_eval/negative_recall"] = recall_score(np.zeros_like(openx_negative_eval_targets), 
                                                                                 np.array(openx_negative_eval_preds) > args.binary_threshold)
                        wandb_eval_log["openx_eval/negative_accuracy"] = accuracy_score(np.zeros_like(openx_negative_eval_targets), 
                                                                                    np.array(openx_negative_eval_preds) > args.binary_threshold)
                    
                    # OpenX Progress Metrics
                    if len(openx_positive_eval_progress_losses) > 0 or len(openx_negative_eval_progress_losses) > 0:
                        wandb_eval_log["openx_eval/progress_loss"] = (np.mean(openx_positive_eval_progress_losses) if len(openx_positive_eval_progress_losses) > 0 else 0) + \
                                                                   (np.mean(openx_negative_eval_progress_losses) if len(openx_negative_eval_progress_losses) > 0 else 0) / 2
                        if len(openx_positive_eval_progress_losses) > 0:
                            wandb_eval_log["openx_eval/positive_progress_loss"] = np.mean(openx_positive_eval_progress_losses)
                        if len(openx_negative_eval_progress_losses) > 0:
                            wandb_eval_log["openx_eval/negative_progress_loss"] = np.mean(openx_negative_eval_progress_losses)
                print("Logging evaluation metrics")
                # wandb.log(wandb_eval_log)

                # Evaluate Extra positive samples
                print("\nEvaluating Extra dataset:")
                if extra_eval_eval_pos_dataloader is not None:
                    print("- Evaluating Extra positive samples")
                    for data in extra_eval_eval_pos_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions

                        progress_pred, class_pred = self_attention_model(video_array, text_array)
                        target = torch.ones(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        extra_eval_eval_pos_losses.append(loss.item())
                        extra_eval_eval_pos_preds.extend(class_pred.squeeze().cpu().numpy())
                        extra_eval_eval_pos_targets.extend(target.cpu().numpy())
                        
                        # Progress loss
                        if args.catagorical_progress:
                            assert "not supported yet"
                        else:
                            progress_loss = progress_loss_function(progress_pred[:,1:].squeeze(), progress_target[:,1:])
                        
                        extra_eval_eval_pos_progress_losses.append(progress_loss.item())
                        extra_eval_eval_pos_progress_preds.extend(progress_pred.squeeze().cpu().numpy())
                        extra_eval_eval_pos_progress_targets.extend(progress_target.cpu().numpy())

                # Evaluate Extra negative samples
                if extra_eval_eval_neg_dataloader is not None:
                    print("- Evaluating Extra negative samples")
                    for data in extra_eval_eval_neg_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions

                        progress_pred, class_pred = self_attention_model(video_array, text_array)
                        target = torch.zeros(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        extra_eval_eval_neg_losses.append(loss.item())
                        extra_eval_eval_neg_preds.extend(class_pred.squeeze().cpu().numpy())
                        extra_eval_eval_neg_targets.extend(target.cpu().numpy())
                        


                # Log Extra metrics if we have data
                if len(extra_eval_eval_pos_preds) > 0 or len(extra_eval_eval_neg_preds) > 0:
                    # Extra Combined Classification Metrics
                    extra_preds = np.array(extra_eval_eval_pos_preds + extra_eval_eval_neg_preds)
                    extra_targets = np.array(extra_eval_eval_pos_targets + extra_eval_eval_neg_targets)
                    
                    wandb_eval_log["extra_eval/loss"] = (np.mean(extra_eval_eval_pos_losses) if len(extra_eval_eval_pos_losses) > 0 else 0) + \
                                                      (np.mean(extra_eval_eval_neg_losses) if len(extra_eval_eval_neg_losses) > 0 else 0) / 2
                    
                    if len(extra_preds) > 0:
                        wandb_eval_log["extra_eval/f1"] = f1_score(extra_targets > args.binary_threshold, extra_preds > args.binary_threshold)
                        wandb_eval_log["extra_eval/precision"] = precision_score(extra_targets > args.binary_threshold, extra_preds > args.binary_threshold)
                        wandb_eval_log["extra_eval/recall"] = recall_score(extra_targets > args.binary_threshold, extra_preds > args.binary_threshold)
                        wandb_eval_log["extra_eval/accuracy"] = accuracy_score(extra_targets > args.binary_threshold, extra_preds > args.binary_threshold)
                    
                    # Extra Positive Sample Metrics
                    if len(extra_eval_eval_pos_preds) > 0:
                        wandb_eval_log["extra_eval/positive_loss"] = np.mean(extra_eval_eval_pos_losses)
                        wandb_eval_log["extra_eval/positive_f1"] = f1_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                         np.array(extra_eval_eval_pos_preds) > args.binary_threshold)
                        wandb_eval_log["extra_eval/positive_precision"] = precision_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                                      np.array(extra_eval_eval_pos_preds) > args.binary_threshold)
                        wandb_eval_log["extra_eval/positive_recall"] = recall_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                                np.array(extra_eval_eval_pos_preds) > args.binary_threshold)
                        wandb_eval_log["extra_eval/positive_accuracy"] = accuracy_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                                    np.array(extra_eval_eval_pos_preds) > args.binary_threshold)
                    
                    # Extra Negative Sample Metrics
                    if len(extra_eval_eval_neg_preds) > 0:
                        wandb_eval_log["extra_eval/negative_loss"] = np.mean(extra_eval_eval_neg_losses)
                        wandb_eval_log["extra_eval/negative_f1"] = f1_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                         np.array(extra_eval_eval_neg_preds) > args.binary_threshold)
                        wandb_eval_log["extra_eval/negative_precision"] = precision_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                                      np.array(extra_eval_eval_neg_preds) > args.binary_threshold)
                        wandb_eval_log["extra_eval/negative_recall"] = recall_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                                np.array(extra_eval_eval_neg_preds) > args.binary_threshold)
                        wandb_eval_log["extra_eval/negative_accuracy"] = accuracy_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                                    np.array(extra_eval_eval_neg_preds) > args.binary_threshold)
                    
                    # Extra Progress Metrics
                    if len(extra_eval_eval_pos_progress_losses) > 0 or len(extra_eval_eval_neg_progress_losses) > 0:
                        wandb_eval_log["extra_eval/progress_loss"] = (np.mean(extra_eval_eval_pos_progress_losses) if len(extra_eval_eval_pos_progress_losses) > 0 else 0) + \
                                                                   (np.mean(extra_eval_eval_neg_progress_losses) if len(extra_eval_eval_neg_progress_losses) > 0 else 0) / 2
                        if len(extra_eval_eval_pos_progress_losses) > 0:
                            wandb_eval_log["extra_eval/positive_progress_loss"] = np.mean(extra_eval_eval_pos_progress_losses)
                        if len(extra_eval_eval_neg_progress_losses) > 0:
                            wandb_eval_log["extra_eval/negative_progress_loss"] = np.mean(extra_eval_eval_neg_progress_losses)

                print("Logging evaluation metrics")
                wandb.log(wandb_eval_log)


        if epoch % 1 == 0:
            # Plot confusion matrix

            self_attention_model.eval()
            with torch.no_grad():
                if args.extra_data_type == "metaworld":

                    plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, binary_threshold = 0.5)
                    plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, binary_threshold = 0.5)
                    if args.two_step_training:
                        plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, binary_threshold = 0.4)
                        plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, binary_threshold = 0.4)
                        plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train",self_attention_model = self_attention_model, args = args, binary_threshold = 0.3)
                        plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args, binary_threshold = 0.3)
                    plot_progress(h5_train_eval_file, "train", self_attention_model, args)
                    plot_progress(h5_eval_file, "eval", self_attention_model, args)

                    # generate_rewind_data
                    # list all pickle files
                    if epoch % 2 == 0:
                        compute_gif = True
                    else:
                        compute_gif = False

                    compute_metrics_multi(args, self_attention_model, threshold=0.5, compute_gif = compute_gif)
                    compute_metrics_multi(args, self_attention_model, threshold=0.4, compute_gif = compute_gif)
                    compute_metrics_multi(args, self_attention_model, threshold=0.3, compute_gif = compute_gif)
 

                else:

                    plot_progress(h5_train_eval_file, "train", self_attention_model, args)
                    plot_progress(h5_eval_file, "eval", self_attention_model, args)
                    plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train", self_attention_model = self_attention_model, args = args)
                    plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", self_attention_model = self_attention_model, args = args)

            # save model

                
            save_dict = {
                "model": self_attention_model.state_dict(),
                "optimizer": base_optimizer.state_dict(),
                "epoch": epoch,
                "args": args
            }

            save_folder = "saved_models"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, f"epoch_{epoch}.pth")
            torch.save(save_dict, save_path)




                




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
    argparser.add_argument('--worker', type=int, default=4)
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
    argparser.add_argument('--view', type=str, default="side", choices=["side", "top"])
    argparser.add_argument('--extra_data_ratio', type=float, default=0.02)
    argparser.add_argument('--catagorical_progress', action='store_true')

    argparser.add_argument('--text_embedding_model', type=str, default="minilm", choices=["minilm", "liv"])
    argparser.add_argument('--eval_interval', type=int, default=2)
    argparser.add_argument('--binary_threshold', type=float, default=0.5)
    argparser.add_argument('--rewind_ratio', type=float, default=0.5)
    argparser.add_argument('--progress_loss_weight', type=float, default=1)
    argparser.add_argument('--weighted_mse', action='store_true')
    argparser.add_argument('--last_frame_pe', action='store_true')



    args = argparser.parse_args()
    main(args)

