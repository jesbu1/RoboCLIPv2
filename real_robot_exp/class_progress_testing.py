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
# from eval_confusion_matrix_dot_product import plot_confusion_matrix
from eval_progress_dot_product import plot_progress
from eval_raw_video_progress import real_video_plot
from utils_clean_dot_product import CosineWithMinLRScheduler, eval_model
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import math
from datetime import date
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt

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
def shorten_name(name, separator=" ", max_length=5):
    parts = name.split(separator)
    return separator.join([part[:max_length] for part in parts])

def plot_matrix_as_image(matrix, names, set, text, prob = False):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.1, len(matrix)))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    shortened_text = [shorten_name(name, max_length = 4) for name in text]
    shortened_names = [shorten_name(name, separator = "-", max_length = 4) for name in names]

    # Label each row and column with the given names
    ax.set_xticklabels(shortened_text, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(shortened_names, fontsize=10)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=12)
# keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    if prob:
        wandb.log({f"confusion_matrix_prob/{set}_prob_confusion_matrix": wandb.Image(fig)})
    else:
        wandb.log({f"confusion_matrix/{set}_confusion_matrix": wandb.Image(fig)})
    # plt.savefig(f"confusion_matrix_{set}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    
def padding_video(video_frames, max_length):
    video_length = len(video_frames)
    if type(video_frames) == np.ndarray:
        video_frames = torch.tensor(video_frames)
    if video_length < max_length:
        # padding first frame
        padding_length = max_length - video_length
        first_frame = video_frames[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_length, 1)
        video_frames = torch.cat([padding_frames, video_frames], dim=0)
    
    elif video_length > max_length:
        frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
        video_frames = video_frames[frame_idx]

    return video_frames

def plot_confusion_matrix(h5_file, set, model, args, pca_text_model = None, pca_video_model = None):
    device = next(model.parameters()).device

    keys = list(h5_file.keys())
    eval_envs = keys


    text_embeddings = []
    text_list = []
    for key in eval_envs:
        embedding = np.asarray(h5_file[key]["minilm_lang_embedding"])[0].reshape(1, -1)
        text_embeddings.append(embedding)
        text_list.append(key)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    text_embeddings = torch.from_numpy(text_embeddings).to(device).float()
    if args.pca:
        text_embeddings = pca_text_model(text_embeddings)

    # if args.normalize_embedding:
    #     text_embeddings = normalize_embeddings(text_embeddings)

    predicted_progress_row = []

    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        video_embedding = np.asarray(h5_file[env]["4"])
        video_embedding = torch.from_numpy(video_embedding).to(device).float()
        if args.subsample_video:
            video_embedding = padding_video(video_embedding, args.max_length)
        if args.pca:
            video_embedding = pca_video_model(video_embedding)

        # if args.normalize_embedding:
        #     traj_data = normalize_embeddings(video_embedding)
        # else:
        traj_data = video_embedding

        traj_data = traj_data.unsqueeze(0)
        triangle_mask = torch.tril(torch.ones(traj_data.shape[1], traj_data.shape[1])).to(device).unsqueeze(0).unsqueeze(0).repeat(traj_data.shape[0], 1, 1, 1)
        
        traj_data = traj_data.repeat(len(text_embeddings),1,1,)
        
        progress_pred, class_pred = model(traj_data, text_embeddings)
        progress_pred = progress_pred[:, -1, :]
        print(progress_pred, class_pred)
        class_pred = (class_pred > 0.5).float()
        progress = progress_pred * class_pred # set to 0 if notin right class
        # progress = progress_pred
        progress = progress.cpu().squeeze().numpy()

        predicted_progress_row.append(progress)
    img = plot_matrix_as_image(predicted_progress_row, eval_envs, set, text_list, prob = False)    
    




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

def plot_progress_predictions(predictions, targets, title, step):
    """Plot predicted vs actual progress values"""
    plt.figure(figsize=(10, 6))
    plt.plot(predictions.cpu().numpy(), label='Predicted Progress', alpha=0.7)
    plt.plot(targets.cpu().numpy(), label='Actual Progress', alpha=0.7)
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to wandb
    wandb.log({f"progress_plot/{title}": wandb.Image(plt)}, step=step)
    plt.close()

def plot_confusion_matrix_custom(predictions, targets, class_names, title, step):
    """Plot confusion matrix for classification results"""
    # Convert predictions to binary
    binary_preds = (predictions >= 0.5).float().cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Compute confusion matrix
    cm = np.zeros((len(class_names), len(class_names)))
    for i in range(len(targets)):
        cm[int(targets[i])][int(binary_preds[i])] += 1
    
    # Normalize confusion matrix
    cm = cm / cm.sum(axis=1, keepdims=True)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]:.2f}',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save to wandb
    wandb.log({f"confusion_matrix/{title}": wandb.Image(plt)}, step=step)
    plt.close()

class ClassProgressTransformer(nn.Module):
    def __init__(self, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Position embeddings for video sequence
        self.pos_embed = nn.Parameter(torch.randn(1, 32, hidden_dim))  # 32 is max_length
        
        # Class token embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Progress prediction head (applied to each frame)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head (applied to class token)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video_frames, text_embed, attention_mask=None):
        batch_size = video_frames.shape[0]
        seq_len = video_frames.shape[1]
        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Add positional embeddings to video
        video_embed = video_embed + self.pos_embed[:, :seq_len, :]
        
        # Expand class token for batch
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        
        # Combine sequence: [class_token, video_frames, text]
        sequence = torch.cat([class_tokens, video_embed, text_embed], dim=1)
        
        # Create attention mask if needed
        if attention_mask is not None:
            # Add mask positions for class token and text token
            extended_mask = torch.ones((batch_size, 2), device=attention_mask.device)  # class token + text token
            attention_mask = torch.cat([extended_mask, attention_mask], dim=1)
        
        # Pass through transformer
        transformed = self.transformer(sequence, src_key_padding_mask=attention_mask if attention_mask is not None else None)
        
        # Get class prediction from class token
        class_pred = self.classification_head(transformed[:, 0])  # Use class token
        
        # Get progress predictions for each frame
        progress_preds = self.progress_head(transformed[:, 1:-1])  # Exclude class token and text token
        
        return progress_preds, class_pred

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


    class_progress_transformer = ClassProgressTransformer(
        video_dim=768,  # Original video embedding dimension
        text_dim=384,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)

    print(class_progress_transformer)
    if args.cosine_scheduler:
        optimizer = torch.optim.AdamW(list(class_progress_transformer.parameters()), 
                                    lr=args.lr, 
                                    weight_decay=0.01)  
        scheduler = CosineWithMinLRScheduler(optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)
    else:
        optimizer = torch.optim.AdamW(list(class_progress_transformer.parameters()), 
                                    lr=args.lr, 
                                    weight_decay=0.01)  
        scheduler = None

    triangular_mask = torch.tril(torch.ones(args.max_length, args.max_length)).to(device).unsqueeze(0).unsqueeze(0)


    for epoch in range(args.epochs):

        class_progress_transformer.train()

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
                progress_pred, class_pred = class_progress_transformer(video_embedding, text_array)
                openx_pred = class_pred[:openx_len]
                extra_pred = class_pred[openx_len:]

                # Calculate focal loss to handle class imbalance
                openx_loss = focal_loss(openx_pred, openx_target)
                extra_loss = focal_loss(extra_pred, extra_target)

                # Add progress prediction loss if applicable
                if args.catagorical_progress:
                    progress_loss = progress_loss_function(progress_pred.view(-1, args.num_classes).squeeze(-1), progress.long())
                else:
                    progress_loss = progress_loss_function(progress_pred.squeeze(), progress)

                # Add L2 regularization loss
                l2_lambda = 0.01  # L2 regularization strength
                l2_reg = torch.tensor(0., requires_grad=True).to(device)
                for param in class_progress_transformer.parameters():
                    l2_reg = l2_reg + torch.norm(param)
                
                # Combined loss with regularization
                loss = openx_loss + extra_loss + progress_loss + l2_lambda * l2_reg

                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(class_progress_transformer.parameters(), max_norm=1.0)
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
     


        # Evaluation
        if epoch % args.eval_interval == 0:  # Only evaluate at specified intervals
            print(f"\nRunning evaluation at epoch {epoch}")
            with torch.no_grad():
                class_progress_transformer.eval()
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
                openx_positive_eval_progress_preds = []
                openx_positive_eval_progress_targets = []
                
                openx_negative_eval_losses = []
                openx_negative_eval_preds = []
                openx_negative_eval_targets = []
                openx_negative_eval_progress_losses = []
                openx_negative_eval_progress_preds = []
                openx_negative_eval_progress_targets = []
                
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
                extra_eval_eval_neg_progress_preds = []
                extra_eval_eval_neg_progress_targets = []

                # Evaluate OpenX positive samples
                if openx_positive_eval_dataloader is not None:
                    for data in openx_positive_eval_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions
                        progress_pred, class_pred = class_progress_transformer(video_array, text_array)
                        target = torch.ones(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        openx_positive_eval_losses.append(loss.item())
                        openx_positive_eval_preds.extend(class_pred.squeeze().cpu().numpy())
                        openx_positive_eval_targets.extend(target.cpu().numpy())
                        
                        # Progress loss
                        if args.catagorical_progress:
                            progress_loss = progress_loss_function(progress_pred.view(-1, args.num_classes).squeeze(-1), progress_target.long())
                        else:
                            progress_loss = progress_loss_function(progress_pred.squeeze(), progress_target)
                        
                        openx_positive_eval_progress_losses.append(progress_loss.item())
                        
                        # Store progress predictions
                        progress_pred_np = progress_pred.detach().cpu().numpy()
                        progress_target_np = progress_target.cpu().numpy()
                        
                        # Only store predictions for actual frames (not padding)
                        valid_mask = ~torch.isnan(progress_target).cpu().numpy()
                        if valid_mask.any():
                            openx_positive_eval_progress_preds.extend(progress_pred_np[valid_mask])
                            openx_positive_eval_progress_targets.extend(progress_target_np[valid_mask])

                # Evaluate OpenX negative samples
                if openx_negative_eval_dataloader is not None:
                    for data in openx_negative_eval_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions
                        progress_pred, class_pred = class_progress_transformer(video_array, text_array)
                        target = torch.zeros(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        openx_negative_eval_losses.append(loss.item())
                        openx_negative_eval_preds.extend(class_pred.squeeze().cpu().numpy())
                        openx_negative_eval_targets.extend(target.cpu().numpy())
                        
                        # Progress loss
                        if args.catagorical_progress:
                            progress_loss = progress_loss_function(progress_pred.view(-1, args.num_classes).squeeze(-1), progress_target.long())
                        else:
                            progress_loss = progress_loss_function(progress_pred.squeeze(), progress_target)
                        
                        openx_negative_eval_progress_losses.append(progress_loss.item())
                        openx_negative_eval_progress_preds.extend(progress_pred.squeeze().cpu().numpy())
                        openx_negative_eval_progress_targets.extend(progress_target.cpu().numpy())
                # Log metrics only if we have data
                if len(openx_positive_eval_preds) > 0 or len(openx_negative_eval_preds) > 0:
                    # OpenX Combined Classification Metrics
                    openx_preds = np.array(openx_positive_eval_preds + openx_negative_eval_preds)
                    openx_targets = np.array(openx_positive_eval_targets + openx_negative_eval_targets)
                    
                    wandb_eval_log["openx_eval/loss"] = (np.mean(openx_positive_eval_losses) if len(openx_positive_eval_losses) > 0 else 0) + \
                                                      (np.mean(openx_negative_eval_losses) if len(openx_negative_eval_losses) > 0 else 0) / 2
                    
                    if len(openx_preds) > 0:
                        wandb_eval_log["openx_eval/f1"] = f1_score(openx_targets > 0.5, openx_preds > 0.5)
                        wandb_eval_log["openx_eval/precision"] = precision_score(openx_targets > 0.5, openx_preds > 0.5)
                        wandb_eval_log["openx_eval/recall"] = recall_score(openx_targets > 0.5, openx_preds > 0.5)
                        wandb_eval_log["openx_eval/accuracy"] = accuracy_score(openx_targets > 0.5, openx_preds > 0.5)
                    
                    # OpenX Positive Sample Metrics
                    if len(openx_positive_eval_preds) > 0:
                        wandb_eval_log["openx_eval/positive_loss"] = np.mean(openx_positive_eval_losses)
                        wandb_eval_log["openx_eval/positive_f1"] = f1_score(np.ones_like(openx_positive_eval_targets), 
                                                                          np.array(openx_positive_eval_preds) > 0.5)
                        wandb_eval_log["openx_eval/positive_precision"] = precision_score(np.ones_like(openx_positive_eval_targets), 
                                                                                       np.array(openx_positive_eval_preds) > 0.5)
                        wandb_eval_log["openx_eval/positive_recall"] = recall_score(np.ones_like(openx_positive_eval_targets), 
                                                                                 np.array(openx_positive_eval_preds) > 0.5)
                        wandb_eval_log["openx_eval/positive_accuracy"] = accuracy_score(np.ones_like(openx_positive_eval_targets), 
                                                                                    np.array(openx_positive_eval_preds) > 0.5)
                    
                    # OpenX Negative Sample Metrics
                    if len(openx_negative_eval_preds) > 0:
                        wandb_eval_log["openx_eval/negative_loss"] = np.mean(openx_negative_eval_losses)
                        wandb_eval_log["openx_eval/negative_f1"] = f1_score(np.zeros_like(openx_negative_eval_targets), 
                                                                          np.array(openx_negative_eval_preds) > 0.5)
                        wandb_eval_log["openx_eval/negative_precision"] = precision_score(np.zeros_like(openx_negative_eval_targets), 
                                                                                       np.array(openx_negative_eval_preds) > 0.5)
                        wandb_eval_log["openx_eval/negative_recall"] = recall_score(np.zeros_like(openx_negative_eval_targets), 
                                                                                 np.array(openx_negative_eval_preds) > 0.5)
                        wandb_eval_log["openx_eval/negative_accuracy"] = accuracy_score(np.zeros_like(openx_negative_eval_targets), 
                                                                                    np.array(openx_negative_eval_preds) > 0.5)
                    
                    # OpenX Progress Metrics
                    if len(openx_positive_eval_progress_losses) > 0 or len(openx_negative_eval_progress_losses) > 0:
                        wandb_eval_log["openx_eval/progress_loss"] = (np.mean(openx_positive_eval_progress_losses) if len(openx_positive_eval_progress_losses) > 0 else 0) + \
                                                                   (np.mean(openx_negative_eval_progress_losses) if len(openx_negative_eval_progress_losses) > 0 else 0) / 2
                        if len(openx_positive_eval_progress_losses) > 0:
                            wandb_eval_log["openx_eval/positive_progress_loss"] = np.mean(openx_positive_eval_progress_losses)
                        if len(openx_negative_eval_progress_losses) > 0:
                            wandb_eval_log["openx_eval/negative_progress_loss"] = np.mean(openx_negative_eval_progress_losses)
                print("Logging evaluation metrics")
                wandb.log(wandb_eval_log)
                
                class_progress_transformer.train()
            
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

                # Evaluate Extra positive samples
                print("\nEvaluating Extra dataset:")
                if extra_eval_eval_pos_dataloader is not None:
                    print("- Evaluating Extra positive samples")
                    for data in extra_eval_eval_pos_dataloader:
                        video_array = data["video_array"].to(device).float()
                        text_array = data["text_array"].squeeze(1).to(device).float()
                        progress_target = data["progress"].to(device).float()
                        
                        # Get predictions
                        progress_pred, class_pred = class_progress_transformer(video_array, text_array)
                        target = torch.ones(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        extra_eval_eval_pos_losses.append(loss.item())
                        extra_eval_eval_pos_preds.extend(class_pred.squeeze().cpu().numpy())
                        extra_eval_eval_pos_targets.extend(target.cpu().numpy())
                        
                        # Progress loss
                        if args.catagorical_progress:
                            progress_loss = progress_loss_function(progress_pred.view(-1, args.num_classes).squeeze(-1), progress_target.long())
                        else:
                            progress_loss = progress_loss_function(progress_pred.squeeze(), progress_target)
                        
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
                        progress_pred, class_pred = class_progress_transformer(video_array, text_array)
                        target = torch.zeros(class_pred.size(0)).to(device)
                        
                        # Classification loss
                        loss = focal_loss(class_pred.squeeze(), target)
                        extra_eval_eval_neg_losses.append(loss.item())
                        extra_eval_eval_neg_preds.extend(class_pred.squeeze().cpu().numpy())
                        extra_eval_eval_neg_targets.extend(target.cpu().numpy())
                        
                        # Progress loss
                        if args.catagorical_progress:
                            progress_loss = progress_loss_function(progress_pred.view(-1, args.num_classes).squeeze(-1), progress_target.long())
                        else:
                            progress_loss = progress_loss_function(progress_pred.squeeze(), progress_target)
                        
                        extra_eval_eval_neg_progress_losses.append(progress_loss.item())
                        extra_eval_eval_neg_progress_preds.extend(progress_pred.squeeze().cpu().numpy())
                        extra_eval_eval_neg_progress_targets.extend(progress_target.cpu().numpy())

                # Log Extra metrics if we have data
                if len(extra_eval_eval_pos_preds) > 0 or len(extra_eval_eval_neg_preds) > 0:
                    # Extra Combined Classification Metrics
                    extra_preds = np.array(extra_eval_eval_pos_preds + extra_eval_eval_neg_preds)
                    extra_targets = np.array(extra_eval_eval_pos_targets + extra_eval_eval_neg_targets)
                    
                    wandb_eval_log["extra_eval/loss"] = (np.mean(extra_eval_eval_pos_losses) if len(extra_eval_eval_pos_losses) > 0 else 0) + \
                                                      (np.mean(extra_eval_eval_neg_losses) if len(extra_eval_eval_neg_losses) > 0 else 0) / 2
                    
                    if len(extra_preds) > 0:
                        wandb_eval_log["extra_eval/f1"] = f1_score(extra_targets > 0.5, extra_preds > 0.5)
                        wandb_eval_log["extra_eval/precision"] = precision_score(extra_targets > 0.5, extra_preds > 0.5)
                        wandb_eval_log["extra_eval/recall"] = recall_score(extra_targets > 0.5, extra_preds > 0.5)
                        wandb_eval_log["extra_eval/accuracy"] = accuracy_score(extra_targets > 0.5, extra_preds > 0.5)
                    
                    # Extra Positive Sample Metrics
                    if len(extra_eval_eval_pos_preds) > 0:
                        wandb_eval_log["extra_eval/positive_loss"] = np.mean(extra_eval_eval_pos_losses)
                        wandb_eval_log["extra_eval/positive_f1"] = f1_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                         np.array(extra_eval_eval_pos_preds) > 0.5)
                        wandb_eval_log["extra_eval/positive_precision"] = precision_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                                      np.array(extra_eval_eval_pos_preds) > 0.5)
                        wandb_eval_log["extra_eval/positive_recall"] = recall_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                                np.array(extra_eval_eval_pos_preds) > 0.5)
                        wandb_eval_log["extra_eval/positive_accuracy"] = accuracy_score(np.ones_like(extra_eval_eval_pos_targets), 
                                                                                    np.array(extra_eval_eval_pos_preds) > 0.5)
                    
                    # Extra Negative Sample Metrics
                    if len(extra_eval_eval_neg_preds) > 0:
                        wandb_eval_log["extra_eval/negative_loss"] = np.mean(extra_eval_eval_neg_losses)
                        wandb_eval_log["extra_eval/negative_f1"] = f1_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                         np.array(extra_eval_eval_neg_preds) > 0.5)
                        wandb_eval_log["extra_eval/negative_precision"] = precision_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                                      np.array(extra_eval_eval_neg_preds) > 0.5)
                        wandb_eval_log["extra_eval/negative_recall"] = recall_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                                np.array(extra_eval_eval_neg_preds) > 0.5)
                        wandb_eval_log["extra_eval/negative_accuracy"] = accuracy_score(np.zeros_like(extra_eval_eval_neg_targets), 
                                                                                    np.array(extra_eval_eval_neg_preds) > 0.5)
                    
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
            
            print("\nLogging metrics to wandb:")
            if len(openx_positive_eval_preds) > 0 or len(openx_negative_eval_preds) > 0:
                print(f"- OpenX metrics: {len(openx_positive_eval_preds)} positive, {len(openx_negative_eval_preds)} negative samples")
            if len(extra_eval_eval_pos_preds) > 0 or len(extra_eval_eval_neg_preds) > 0:
                print(f"- Extra metrics: {len(extra_eval_eval_pos_preds)} positive, {len(extra_eval_eval_neg_preds)} negative samples")
            
            print("Evaluation complete\n")
            
            class_progress_transformer.train()
            
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




        if epoch % 5 == 0:
            class_progress_transformer.eval()
            with torch.no_grad():
                if args.extra_data_type == "metaworld":
                    # plot_progress(h5_train_eval_file, "train", class_progress_transformer, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                    plot_confusion_matrix(h5_file = h5_train_eval_file,
                                        set = "train",
                                        model = class_progress_transformer,
                                        args = args,
                                        pca_text_model = pca_text_model,
                                        pca_video_model = pca_video_model)

                    # plot_progress(h5_eval_file, "eval", class_progress_transformer, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                    plot_confusion_matrix(h5_file = h5_eval_file,
                                        set = "eval",
                                        model = class_progress_transformer,
                                        args = args,
                                        pca_text_model = pca_text_model,
                                        pca_video_model = pca_video_model)               

                else:

                    # plot_progress(h5_train_eval_file, "train", class_progress_transformer, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                    # plot_progress(h5_eval_file, "eval", class_progress_transformer, args, pca_text_model = pca_text_model, pca_video_model = pca_video_model)
                    plot_confusion_matrix(h5_file = h5_train_eval_file,
                                        set = "train",
                                        model = class_progress_transformer,
                                        args = args, 
                                        pca_text_model = pca_text_model, 
                                        pca_video_model = pca_video_model)
                    plot_confusion_matrix(h5_file = h5_eval_file,
                                        set = "eval",
                                        model = class_progress_transformer,
                                        args = args,
                                        pca_text_model = pca_text_model,
                                        pca_video_model = pca_video_model)


            






if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_dino_train.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/home/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/mnt/ssd_a_4tb/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "real_world"], default="real_world")
    argparser.add_argument('--batch_size', type=int, default=512)
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
    argparser.add_argument('--eval_interval', type=int, default=1)

    args = argparser.parse_args()
    main(args)