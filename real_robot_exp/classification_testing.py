import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
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
from models_dot_product import VideoTransformerEncoder
# , RewardOneStepNewPositionEmbeddingPredictor
from eval_confusion_matrix_dot_product import plot_confusion_matrix
from eval_progress_dot_product import plot_progress
from eval_raw_video_progress import real_video_plot
from utils_clean_dot_product import CosineWithMinLRScheduler, eval_model
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import math
from datetime import date
import pickle
import torch.nn as nn


os.environ["TOKENIZERS_PARALLELISM"] = "False"

class PCATransform(torch.nn.Module):
    def __init__(self, components, mean):
        super().__init__()
        self.linear = torch.nn.Linear(components.shape[1], components.shape[0], bias=False)
        self.register_buffer("mean", torch.from_numpy(mean).float())

        # Set weights (PyTorch Linear expects weights transposed)
        # self.linear.weight = torch.nn.Parameter(components)
        self.linear.weight = torch.nn.Parameter(torch.from_numpy(components).float(), requires_grad=False)

    def forward(self, x):
        return self.linear(x - self.mean)

# Move model to GPU





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

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce_loss = F.binary_cross_entropy(pred.squeeze(), target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1-pt)**gamma * bce_loss
    return focal_loss.mean()

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

    if args.pca:
        experiment_name = "PCA_" + experiment_name

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

    if args.layer_norm:
        experiment_name += "_LayerNorm"


    if args.learner_parameter:
        experiment_name += "_LearnerPara"
    if args.cosine_scheduler:
        experiment_name += "_CosScheduler"
    if args.clip_grad:
        experiment_name += "_ClipGrad"
    experiment_name += "_View_" + str(args.view)
    experiment_name += "_ExtraDataRatio_" + str(args.extra_data_ratio)
    
    experiment_name += "_DecoderNum_" + str(args.decoder_num)
    experiment_name += "_epochs_" + str(args.epochs)
    experiment_name += "_lr_" + str(args.lr)
    # experiment_name += "_1_demo"
    experiment_name += "_FIXFIX"
    
    if args.extra_data_type == "metaworld":
        group_name = "MetaWorld"
    else:
        group_name = "RealWorld_Koch"
    # get today date


    
    # group_name += "Feb11th"
    group_name = "DotProduct" + group_name
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=group_name,
        config=args,
        name=experiment_name,
    )



    if args.extra_data_type == "metaworld":
        h5_train_eval_file = h5py.File("metaworld_embedding_5_demo_dataset_v3_train.h5", "r")
        h5_eval_file = h5py.File("metaworld_embedding_5_demo_dataset_v3_eval.h5", "r")
        extra_data_path = "metaworld_embedding_5_demo_dataset_v3_train.h5"
    else:

        if args.view == "side":
            h5_train_eval_file = h5py.File("usc_koch_rewind_reward_side_only_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_reward_side_only_eval.h5", "r")
            extra_data_path = "usc_koch_rewind_reward_side_only_train.h5"
        elif args.view == "top":
            h5_train_eval_file = h5py.File("usc_koch_rewind_reward_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_reward_eval.h5", "r")
            extra_data_path = "usc_koch_rewind_reward_train.h5"
        elif args.view == "both":
            h5_train_eval_file = h5py.File("usc_koch_rewind_dino_reward_side_main_train.h5", "r")
            h5_eval_file = h5py.File("usc_koch_rewind_dino_reward_side_main_eval.h5", "r")
            extra_data_path = "usc_koch_rewind_dino_reward_side_main_train.h5"
    embedding_dim = 1024



    if args.pca:
        pca_video_model_path = "pca_models/pca_video_model_512.pkl"
        pca_text_model_path = "pca_models/pca_text_model_512.pkl"
        pca_video_model_para = pickle.load(open(pca_video_model_path, "rb"))
        pca_text_model_para = pickle.load(open(pca_text_model_path, "rb"))
        embedding_dim = pca_video_model_para.components_.shape[0]
        pca_video_model = PCATransform(pca_video_model_para.components_, pca_video_model_para.mean_)
        pca_text_model = PCATransform(pca_text_model_para.components_, pca_text_model_para.mean_)
        pca_video_model = pca_video_model.to(device).eval()
        pca_text_model = pca_text_model.to(device).eval()


    else:
        pca_video_model = None
        pca_text_model = None


    eval_dataset = None
    eval_dataloader = None
    if args.openx_data:
        openx_dataset = LivRealVideoTrainDataset(args, args.h5_embedding_path, split = False, sample_neg=False)
        if args.extra_data_type == "metaworld":
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        else:
            extra_dataset = LivRealVideoTrainDataset(args, extra_data_path, split = False, sample_neg=True)
        
        openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
        extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

        openx_dataloader = DataLoader(openx_dataset, batch_size=openx_batch_size, shuffle=True, num_workers=int(args.worker * 16), drop_last=True, pin_memory=True)
        extra_dataloader = DataLoader(extra_dataset, batch_size=extra_batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)


        # h5_openx_eval_file = h5py.File("/home/jzhang96/openx_embeddings_test_dataset_progrssed.h5", "r")
        # h5_openx_eval_file = h5py.File("/mnt/ssd_a_4tb/jzhang96/openx_embeddings_test_dataset_progrssed.h5", "r")
        h5_openx_eval_file = h5py.File("/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_dino_test.h5", "r")

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



    extra_eval_eval_pos_dataset = LivRealVideoEvalDataset(args, h5_eval_file, label = "positive", dataset = "extra")
    extra_eval_eval_neg_dataset = LivRealVideoEvalDataset(args, h5_eval_file, label = "negative", dataset = "extra")


    extra_eval_eval_pos_dataloader = DataLoader(extra_eval_eval_pos_dataset, batch_size=5, shuffle=True, num_workers=1, drop_last=True)
    extra_eval_eval_neg_dataloader = DataLoader(extra_eval_eval_neg_dataset, batch_size=5, shuffle=True, num_workers=1, drop_last=True)


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


    class ClassificationMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            # MLP takes concatenation of: first frame + last frame + text embedding
            # So input size is 3 * input_dim
            self.dropout = nn.Dropout(p=0.0)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                # nn.LayerNorm(input_dim * 2),  
                nn.GELU(),
                self.dropout,
                nn.Linear(input_dim // 2, input_dim // 4),
                # nn.LayerNorm(input_dim // 2),
                nn.GELU(),
                self.dropout,
                nn.Linear(input_dim // 4, input_dim // 8),
                # nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                self.dropout,
                nn.Linear(input_dim // 8, 1),
                nn.Sigmoid()
            )
            print(self.mlp)
        
        def forward(self, first_frame_embed, last_frame_embed, text_embed):
            # Concatenate first frame, last frame and text embeddings
            combined = torch.cat([first_frame_embed, last_frame_embed, text_embed], dim=1)
            return self.mlp(combined)


    video_encoder = VideoTransformerEncoder(embedding_dim, args = args).to(device)
    classifier = ClassificationMLP(768 * 2 + 384).to(device)

    print(video_encoder)
    if args.cosine_scheduler:
        optimizer = torch.optim.AdamW(list(video_encoder.parameters()) + list(classifier.parameters()), 
                                    lr=args.lr, 
                                    weight_decay=0.01)  
        scheduler = CosineWithMinLRScheduler(optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)
    else:
        optimizer = torch.optim.AdamW(list(video_encoder.parameters()) + list(classifier.parameters()), 
                                    lr=args.lr, 
                                    weight_decay=0.01)  
        scheduler = None

    triangular_mask = torch.tril(torch.ones(args.max_length, args.max_length)).to(device).unsqueeze(0).unsqueeze(0)


    for epoch in range(args.epochs):

        video_encoder.train()

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
                positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"].squeeze()], dim = 0).to(device).float()              
                positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)
                positive_progress_mask = torch.ones_like(positive_progress).bool()

                negative_video_array_1 = torch.roll(positive_video_array, extra_len, 0)
                negative_text_array_1 = positive_text_array.clone()
                negative_progress_1 = torch.zeros_like(positive_progress)
                negative_progress_mask_1 = torch.ones_like(negative_progress_1).bool()
                stop_idx = round(negative_progress_1.size(1) * args.negative_mask_ratio)
                negative_progress_mask_1[:, :stop_idx] = False


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

                batch_triangular_mask = triangular_mask.repeat(openx_len + extra_len, 1, 1, 1).bool()

                
                # video_embedding = video_encoder(video_array, batch_triangular_mask)
                video_embedding = video_array
                
                # Get first and last frame embeddings
                first_frame_embedding = video_embedding[:, 0, :]  # Shape: [batch_size, embedding_dim]
                last_frame_embedding = video_embedding[:, -1, :]  # Shape: [batch_size, embedding_dim]
                
                # Binary classification targets
                compressed_extra_class_label = extra_data["class_label"][:, 0].float()
                openx_target = torch.cat([torch.ones(openx_len // 2), torch.zeros(openx_len // 2)], dim=0).to(device)
                extra_target = torch.cat([compressed_extra_class_label, torch.zeros(extra_len // 2)], dim=0).to(device)

                # Get predictions from classifier
                openx_pred = classifier(
                    first_frame_embedding[:openx_len], 
                    last_frame_embedding[:openx_len], 
                    text_array[:openx_len]
                )
                extra_pred = classifier(
                    first_frame_embedding[openx_len:],
                    last_frame_embedding[openx_len:],
                    text_array[openx_len:]
                )

                # Calculate focal loss to handle class imbalance
                openx_loss = focal_loss(openx_pred, openx_target)
                extra_loss = focal_loss(extra_pred, extra_target)

                # Add L2 regularization loss
                l2_lambda = 0.01  # L2 regularization strength
                l2_reg = torch.tensor(0., requires_grad=True).to(device)
                for param in classifier.parameters():
                    l2_reg = l2_reg + torch.norm(param)
                
                # Combined loss with regularization
                loss = (1 - args.extra_data_ratio) * openx_loss + args.extra_data_ratio * extra_loss

                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Log all metrics
                wandb_log = {
                    "openx_loss": openx_loss.item(),
                    "extra_loss": extra_loss.item(),
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    # OpenX metrics
                    "openx_accuracy": compute_metrics(openx_pred.squeeze(), openx_target)['accuracy'],
                    "openx_precision": compute_metrics(openx_pred.squeeze(), openx_target)['precision'],
                    "openx_recall": compute_metrics(openx_pred.squeeze(), openx_target)['recall'],
                    "openx_f1": compute_metrics(openx_pred.squeeze(), openx_target)['f1'],
                    # Extra metrics
                    "extra_accuracy": compute_metrics(extra_pred.squeeze(), extra_target)['accuracy'],
                    "extra_precision": compute_metrics(extra_pred.squeeze(), extra_target)['precision'],
                    "extra_recall": compute_metrics(extra_pred.squeeze(), extra_target)['recall'],
                    "extra_f1": compute_metrics(extra_pred.squeeze(), extra_target)['f1'],
                    # Combined metrics (weighted average)
                    "combined_accuracy": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['accuracy'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['accuracy'],
                    "combined_precision": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['precision'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['precision'],
                    "combined_recall": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['recall'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['recall'],
                    "combined_f1": (1 - args.extra_data_ratio) * compute_metrics(openx_pred.squeeze(), openx_target)['f1'] + args.extra_data_ratio * compute_metrics(extra_pred.squeeze(), extra_target)['f1']
                }
                wandb.log(wandb_log)
     


        with torch.no_grad():
            video_encoder.eval()
            classifier.eval()
            wandb_eval_log = {}
            
            if openx_positive_eval_dataloader is not None:
                # Evaluate on positive pairs
                all_pos_preds = []
                all_pos_targets = []
                total_pos_loss = 0
                total_pos_samples = 0
                
                for data in openx_positive_eval_dataloader:
                    video_array = data["video_array"].to(device).float()
                    text_array = data["text_array"].squeeze(1).to(device).float()
                    
                    if pca_video_model is not None:
                        video_array = pca_video_model(video_array)
                    if pca_text_model is not None:
                        text_array = pca_text_model(text_array)
                    
                    batch_triangular_mask = triangular_mask.repeat(video_array.size(0), 1, 1, 1).bool()
                    video_embedding = video_array
                    first_frame_embedding = video_embedding[:, 0, :]
                    last_frame_embedding = video_embedding[:, -1, :]
                    
                    pred = classifier(first_frame_embedding, last_frame_embedding, text_array)
                    target = torch.ones(pred.size(0)).to(device)
                    
                    loss = focal_loss(pred.squeeze(), target)
                    total_pos_loss += loss.item() * pred.size(0)
                    total_pos_samples += pred.size(0)
                    
                    all_pos_preds.append(pred.squeeze())
                    all_pos_targets.append(target)
                
                # Concatenate all predictions and targets
                all_pos_preds = torch.cat(all_pos_preds)
                all_pos_targets = torch.cat(all_pos_targets)
                pos_metrics = compute_metrics(all_pos_preds, all_pos_targets)
                
                wandb_eval_log.update({
                    "openx_eval/positive_loss": total_pos_loss / total_pos_samples,
                    "openx_eval/positive_accuracy": pos_metrics['accuracy'],
                    "openx_eval/positive_precision": pos_metrics['precision'],
                    "openx_eval/positive_recall": pos_metrics['recall'],
                    "openx_eval/positive_f1": pos_metrics['f1']
                })
            
            if openx_negative_eval_dataloader is not None:
                # Evaluate on negative pairs
                all_neg_preds = []
                all_neg_targets = []
                total_neg_loss = 0
                total_neg_samples = 0
                
                for data in openx_negative_eval_dataloader:
                    video_array = data["video_array"].to(device).float()
                    text_array = data["text_array"].squeeze(1).to(device).float()
                    
                    if pca_video_model is not None:
                        video_array = pca_video_model(video_array)
                    if pca_text_model is not None:
                        text_array = pca_text_model(text_array)
                    
                    batch_triangular_mask = triangular_mask.repeat(video_array.size(0), 1, 1, 1).bool()
                    video_embedding = video_array
                    first_frame_embedding = video_embedding[:, 0, :]
                    last_frame_embedding = video_embedding[:, -1, :]
                    
                    pred = classifier(first_frame_embedding, last_frame_embedding, text_array)
                    target = torch.zeros(pred.size(0)).to(device)
                    
                    loss = focal_loss(pred.squeeze(), target)
                    total_neg_loss += loss.item() * pred.size(0)
                    total_neg_samples += pred.size(0)
                    
                    all_neg_preds.append(pred.squeeze())
                    all_neg_targets.append(target)
                
                # Concatenate all predictions and targets
                all_neg_preds = torch.cat(all_neg_preds)
                all_neg_targets = torch.cat(all_neg_targets)
                neg_metrics = compute_metrics(all_neg_preds, all_neg_targets)
                
                wandb_eval_log.update({
                    "openx_eval/negative_loss": total_neg_loss / total_neg_samples,
                    "openx_eval/negative_accuracy": neg_metrics['accuracy'],
                    "openx_eval/negative_precision": neg_metrics['precision'],
                    "openx_eval/negative_recall": neg_metrics['recall'],
                    "openx_eval/negative_f1": neg_metrics['f1']
                })
                
                # Calculate combined metrics for OpenX
                all_preds = torch.cat([all_pos_preds, all_neg_preds])
                all_targets = torch.cat([all_pos_targets, all_neg_targets])
                combined_metrics = compute_metrics(all_preds, all_targets)
                
                wandb_eval_log.update({
                    "openx_eval/combined_accuracy": combined_metrics['accuracy'],
                    "openx_eval/combined_precision": combined_metrics['precision'],
                    "openx_eval/combined_recall": combined_metrics['recall'],
                    "openx_eval/combined_f1": combined_metrics['f1']
                })

            if extra_eval_eval_pos_dataset is not None:
                # Evaluate on positive pairs
                all_pos_preds = []
                all_pos_targets = []
                total_pos_loss = 0
                total_pos_samples = 0
                
                for data in extra_eval_eval_pos_dataloader:
                    video_array = data["video_array"].to(device).float()
                    text_array = data["text_array"].squeeze(1).to(device).float()
                    
                    if pca_video_model is not None:
                        video_array = pca_video_model(video_array)
                    if pca_text_model is not None:
                        text_array = pca_text_model(text_array)
                    
                    batch_triangular_mask = triangular_mask.repeat(video_array.size(0), 1, 1, 1).bool()
                    # video_embedding = video_encoder(video_array, batch_triangular_mask)
                    video_embedding = video_array

                    first_frame_embedding = video_embedding[:, 0, :]
                    last_frame_embedding = video_embedding[:, -1, :]
                    
                    pred = classifier(first_frame_embedding, last_frame_embedding, text_array)
                    target = torch.ones(pred.size(0)).to(device)
                    
                    loss = focal_loss(pred.squeeze(), target)
                    total_pos_loss += loss.item() * pred.size(0)
                    total_pos_samples += pred.size(0)
                    
                    all_pos_preds.append(pred.squeeze())
                    all_pos_targets.append(target)
                
                # Concatenate all predictions and targets
                all_pos_preds = torch.cat(all_pos_preds)
                all_pos_targets = torch.cat(all_pos_targets)
                pos_metrics = compute_metrics(all_pos_preds, all_pos_targets)
                
                wandb_eval_log.update({
                    "extra_eval/positive_loss": total_pos_loss / total_pos_samples,
                    "extra_eval/positive_accuracy": pos_metrics['accuracy'],
                    "extra_eval/positive_precision": pos_metrics['precision'],
                    "extra_eval/positive_recall": pos_metrics['recall'],
                    "extra_eval/positive_f1": pos_metrics['f1']
                })
            
            if extra_eval_eval_neg_dataset is not None:
                # Evaluate on negative pairs
                all_neg_preds = []
                all_neg_targets = []
                total_neg_loss = 0
                total_neg_samples = 0
                
                for data in extra_eval_eval_neg_dataloader:
                    video_array = data["video_array"].to(device).float()
                    text_array = data["text_array"].squeeze(1).to(device).float()
                    
                    if pca_video_model is not None:
                        video_array = pca_video_model(video_array)
                    if pca_text_model is not None:
                        text_array = pca_text_model(text_array)
                    
                    batch_triangular_mask = triangular_mask.repeat(video_array.size(0), 1, 1, 1).bool()
                    # video_embedding = video_encoder(video_array, batch_triangular_mask)
                    video_embedding = video_array

                    first_frame_embedding = video_embedding[:, 0, :]
                    last_frame_embedding = video_embedding[:, -1, :]
                    
                    pred = classifier(first_frame_embedding, last_frame_embedding, text_array)
                    target = torch.zeros(pred.size(0)).to(device)
                    
                    loss = focal_loss(pred.squeeze(), target)
                    total_neg_loss += loss.item() * pred.size(0)
                    total_neg_samples += pred.size(0)
                    
                    all_neg_preds.append(pred.squeeze())
                    all_neg_targets.append(target)
                
                # Concatenate all predictions and targets
                all_neg_preds = torch.cat(all_neg_preds)
                all_neg_targets = torch.cat(all_neg_targets)
                neg_metrics = compute_metrics(all_neg_preds, all_neg_targets)
                
                wandb_eval_log.update({
                    "extra_eval/negative_loss": total_neg_loss / total_neg_samples,
                    "extra_eval/negative_accuracy": neg_metrics['accuracy'],
                    "extra_eval/negative_precision": neg_metrics['precision'],
                    "extra_eval/negative_recall": neg_metrics['recall'],
                    "extra_eval/negative_f1": neg_metrics['f1']
                })
                
                # Calculate combined metrics
                all_preds = torch.cat([all_pos_preds, all_neg_preds])
                all_targets = torch.cat([all_pos_targets, all_neg_targets])
                combined_metrics = compute_metrics(all_preds, all_targets)
                
                wandb_eval_log.update({
                    "extra_eval/combined_accuracy": combined_metrics['accuracy'],
                    "extra_eval/combined_precision": combined_metrics['precision'],
                    "extra_eval/combined_recall": combined_metrics['recall'],
                    "extra_eval/combined_f1": combined_metrics['f1']
                })

            wandb.log(wandb_eval_log)
            
            # if epoch % 5 == 0:
            #     video_encoder.eval()
                # with torch.no_grad():
                #     if args.extra_data_type == "metaworld":
                #         plot_progress(h5_train_eval_file, "train", video_encoder, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                #         plot_confusion_matrix(h5_file = h5_train_eval_file,
                #                             set = "train",
                #                             video_encoder = video_encoder,
                #                             args = args,
                #                             pca_text_model = pca_text_model,
                #                             pca_video_model = pca_video_model)

                #         plot_progress(h5_eval_file, "eval", video_encoder, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                #         plot_confusion_matrix(h5_file = h5_eval_file,
                #                             set = "eval",
                #                             video_encoder = video_encoder,
                #                             args = args,
                #                             pca_text_model = pca_text_model,
                #                             pca_video_model = pca_video_model)               

                    # else:

                    #     # plot_progress(h5_train_eval_file, "train", video_encoder, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                    #     # plot_progress(h5_eval_file, "eval", video_encoder, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                    #     plot_confusion_matrix(h5_file = h5_train_eval_file,
                    #                         set = "train",
                    #                         video_encoder = video_encoder,
                    #                         args = args, 
                    #                         pca_text_model = pca_text_model, 
                    #                         pca_video_model = pca_video_model)
                    #     plot_confusion_matrix(h5_file = h5_eval_file,
                    #                         set = "eval",
                    #                         video_encoder = video_encoder,
                    #                         args = args,
                    #                         pca_text_model = pca_text_model,
                    #                         pca_video_model = pca_video_model)





            # if epoch % 20 == 0:
            #     save_path = "/home/jzhang96/roboclip_v2_models"
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     save_path = os.path.join(save_path, experiment_name)
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     save_path = os.path.join(save_path, "model_" + str(epoch) + ".pth")
            #     save_dict = {
            #         "model": self_attention_model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "epoch": epoch,
            #         "args": args
            #     }
            #     torch.save(save_dict, save_path)







            






if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_dino_train.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/home/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/mnt/ssd_a_4tb/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "real_world"], default="real_world")
    argparser.add_argument('--batch_size', type=int, default=1024)
    argparser.add_argument('--catagorical_progress', action='store_true')
    argparser.add_argument('--two_step_training', action='store_true')
    argparser.add_argument('--epochs', type=int, default=10000)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--rewind', action='store_true')
    argparser.add_argument('--normalize_embedding', action='store_true')    
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--max_length', type=int, default=32)
    argparser.add_argument('--positional_encoding', action='store_true')
    argparser.add_argument('--openx_data', action='store_true')
    argparser.add_argument('--layer_norm', action='store_true')
    argparser.add_argument('--decoder_num', type=int, default=1)
    argparser.add_argument('--learner_parameter', action='store_true')
    argparser.add_argument('--cosine_scheduler', action='store_true')
    argparser.add_argument('--clip_grad', action='store_true')
    argparser.add_argument('--view', type=str, default="side", choices=["side", "top", "both"])
    argparser.add_argument('--extra_data_ratio', type=float, default=0.8)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--norm_length', action='store_true')
    argparser.add_argument('--negative_mask_ratio', type=float, default=0.8)

    args = argparser.parse_args()
    main(args)






    
    