import torch
# from dataloader_liv_decoder_5_demo import video_collate_triangular_fn, LivVideoDecoderDataset5Frames
from dataset_clean_dot_token import LivRealVideoTrainTokenDataset, LivRealVideoEvalTokenDataset, VideoTextTokenCollateFn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import wandb
from tqdm import tqdm
import h5py
from torch.nn.functional import mse_loss
import os
from models_dot_product import VideoTransformerEncoder, TextTransformerEncoder
# , RewardOneStepNewPositionEmbeddingPredictor
from eval_confusion_matrix_dot_product_token import plot_confusion_matrix
from eval_progress_dot_product_token import plot_progress
from eval_raw_video_progress import real_video_plot
from utils_clean_dot_product import CosineWithMinLRScheduler, eval_model_token
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import math
from datetime import date
import pickle


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

    if args.roll_percentage != 1.0:
        experiment_name = "RollPercentage_" + str(args.roll_percentage) + "_" + experiment_name

    if args.pca:
        experiment_name = "PCA_" + experiment_name

    if args.openx_data:
        experiment_name += "_AddOpenXData"
    


    experiment_name += "_heads_" + str(args.attention_heads)


    if args.rewind:
        experiment_name += "_ReWind"
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
    group_name = "TokenDotProduct" + group_name
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
        openx_dataset = LivRealVideoTrainTokenDataset(args, args.h5_embedding_path, split = False, sample_neg=False)
        if args.extra_data_type == "metaworld":
            extra_dataset = LivRealVideoTrainTokenDataset(args, extra_data_path, split = False, sample_neg=True)
        else:
            extra_dataset = LivRealVideoTrainTokenDataset(args, extra_data_path, split = False, sample_neg=True)
        
        openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
        extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

        openx_dataloader = DataLoader(openx_dataset, 
                                      batch_size=openx_batch_size, 
                                      shuffle=True, 
                                      num_workers=int(args.worker * 16), 
                                      drop_last=True, 
                                      pin_memory=True, 
                                      collate_fn=VideoTextTokenCollateFn)
        extra_dataloader = DataLoader(extra_dataset, 
                                      batch_size=extra_batch_size, 
                                      shuffle=True, 
                                      num_workers=args.worker, 
                                      drop_last=True, 
                                      pin_memory=True,
                                     collate_fn=VideoTextTokenCollateFn)


        # h5_openx_eval_file = h5py.File("/home/jzhang96/openx_embeddings_test_dataset_progrssed.h5", "r")
        # h5_openx_eval_file = h5py.File("/mnt/ssd_a_4tb/jzhang96/openx_embeddings_test_dataset_progrssed.h5", "r")
        h5_openx_eval_file = h5py.File("/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5", "r")

        positive_eval_openx_dataset = LivRealVideoEvalTokenDataset(args, 
                                                        h5_openx_eval_file,
                                                        label = "positive",
                                                        )
        negative_eval_openx_dataset = LivRealVideoEvalTokenDataset(args,
                                                        h5_openx_eval_file,
                                                        label = "negative")
        
        openx_positive_eval_dataloader = DataLoader(positive_eval_openx_dataset, 
                                                    batch_size=args.batch_size // 8, 
                                                    shuffle=True, 
                                                    num_workers=2, 
                                                    drop_last=True, 
                                                    pin_memory=True,
                                                    collate_fn=VideoTextTokenCollateFn)
        openx_negative_eval_dataloader = DataLoader(negative_eval_openx_dataset, 
                                                    batch_size=args.batch_size // 8, 
                                                    shuffle=True, 
                                                    num_workers=2, 
                                                    drop_last=True, 
                                                    pin_memory=True,
                                                    collate_fn=VideoTextTokenCollateFn)




    extra_eval_eval_pos_dataset = LivRealVideoEvalTokenDataset(args, 
                                                               h5_eval_file, 
                                                               label = "positive", 
                                                               )
    extra_eval_eval_neg_dataset = LivRealVideoEvalTokenDataset(args, 
                                                               h5_eval_file, 
                                                               label = "negative", 
                                                               )


    extra_eval_eval_pos_dataloader = DataLoader(extra_eval_eval_pos_dataset, 
                                                batch_size=5, 
                                                shuffle=True, 
                                                num_workers=1, 
                                                drop_last=True,
                                                pin_memory=True,
                                                collate_fn=VideoTextTokenCollateFn)
    extra_eval_eval_neg_dataloader = DataLoader(extra_eval_eval_neg_dataset, 
                                                batch_size=5, 
                                                shuffle=True, 
                                                num_workers=1, 
                                                drop_last=True,
                                                pin_memory=True,
                                                collate_fn=VideoTextTokenCollateFn)



    progress_loss_function = mse_loss

    video_encoder = VideoTransformerEncoder(embedding_dim, args = args).to(device)
    text_encoder = TextTransformerEncoder(emb_size = embedding_dim,max_t=70, num_heads=args.attention_heads, num_layers=2, ff_dim=1024).to(device)


    print(video_encoder)
    if args.cosine_scheduler:
        # optimize together
        optimizer = torch.optim.Adam([{"params": video_encoder.parameters()}, {"params": text_encoder.parameters()}], lr=args.lr)
        scheduler = CosineWithMinLRScheduler(optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)
    else:
        optimizer = torch.optim.Adam([{"params": video_encoder.parameters()}, {"params": text_encoder.parameters()}], lr=args.lr)
        scheduler = None

    triangular_mask = torch.tril(torch.ones(args.max_length, args.max_length)).to(device).unsqueeze(0).unsqueeze(0)

    for epoch in range(args.epochs):

        video_encoder.train()
        text_encoder.train()

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

                openx_len = openx_data["video_array"].size(0)
                extra_len = extra_data["video_array"].size(0)

                openx_text = openx_data["text_array"].to(device).float()
                extra_text = extra_data["text_array"].to(device).float()
                openx_text_mask = openx_data["text_mask"].to(device).float()
                extra_text_mask = extra_data["text_mask"].to(device).float()
                more_len = max(openx_text.size(1), extra_text.size(1)) - min(openx_text.size(1), extra_text.size(1))
                if openx_text.size(1) < extra_text.size(1):
                    openx_text = F.pad(openx_text, (0, 0, 0, more_len), "constant", 0)
                    openx_text_mask = F.pad(openx_text_mask, (0, more_len), "constant", 1)
                else:
                    extra_text = F.pad(extra_text, (0, 0, 0, more_len), "constant", 0)
                    extra_text_mask = F.pad(extra_text_mask, (0, more_len), "constant", 1)

                positive_video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
                positive_text_array = torch.cat([openx_text, extra_text], dim = 0).to(device).float()  
                positive_text_mask = torch.cat([openx_text_mask, extra_text_mask], dim = 0).to(device).float()         
                positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)
                positive_progress_mask = torch.ones_like(positive_progress).bool()


                negative_video_array_1 = torch.roll(positive_video_array, extra_len, 0)
                select_length = int(round(positive_video_array.size(0) * args.roll_percentage))
                neg_openx_video_array = negative_video_array_1[:openx_len]
                rand_idx = torch.randperm(neg_openx_video_array.size(0))
                neg_openx_video_array = neg_openx_video_array[rand_idx][:select_length]
                negative_text_array_1 = positive_text_array.clone()[rand_idx][:select_length]
                negative_text_mask_1 = positive_text_mask.clone()[rand_idx][:select_length]
                negative_progress_1 = torch.zeros_like(positive_progress)[:neg_openx_video_array.size(0)][:select_length]
                
                negative_progress_mask_1 = torch.ones_like(negative_progress_1).bool()[:select_length]
                stop_idx = round(negative_progress_1.size(1) * args.negative_mask_ratio)
                negative_progress_mask_1[:, :stop_idx] = False

                neg_extra_video_array = negative_video_array_1[openx_len:]
                negative_extra_text_array_1 = positive_text_array.clone()[openx_len:]
                negative_extra_text_mask_1 = positive_text_mask.clone()[openx_len:]
                negative_extra_progress_1 = torch.zeros_like(positive_progress)[openx_len:]
                negative_extra_progress_mask_1 = torch.ones_like(negative_extra_progress_1).bool()
                stop_idx = round(negative_extra_progress_1.size(1) * args.negative_mask_ratio)
                negative_extra_progress_mask_1[:, :stop_idx] = False




                openx_pos_video_array = torch.cat([positive_video_array[:openx_len], neg_openx_video_array], dim = 0)
                openx_pos_text_array = torch.cat([positive_text_array[:openx_len], negative_text_array_1], dim = 0)
                openx_pos_text_mask = torch.cat([positive_text_mask[:openx_len], negative_text_mask_1], dim = 0)
                openx_pos_progress = torch.cat([positive_progress[:openx_len], negative_progress_1], dim = 0)
                openx_pos_progress_mask = torch.cat([positive_progress_mask[:openx_len], negative_progress_mask_1], dim = 0)

                extra_pos_video_array = torch.cat([positive_video_array[openx_len:], neg_extra_video_array], dim = 0)
                extra_pos_text_array = torch.cat([positive_text_array[openx_len:], negative_extra_text_array_1], dim = 0)
                extra_pos_text_mask = torch.cat([positive_text_mask[openx_len:], negative_extra_text_mask_1], dim = 0)
                extra_pos_progress = torch.cat([positive_progress[openx_len:], negative_extra_progress_1], dim = 0)
                extra_pos_progress_mask = torch.cat([positive_progress_mask[openx_len:], negative_extra_progress_mask_1], dim = 0)

                video_array = torch.cat([openx_pos_video_array, extra_pos_video_array], dim = 0)
                text_array = torch.cat([openx_pos_text_array, extra_pos_text_array], dim = 0)
                text_mask = torch.cat([openx_pos_text_mask, extra_pos_text_mask], dim = 0)
                progress = torch.cat([openx_pos_progress, extra_pos_progress], dim = 0).float()
                progress_mask = torch.cat([openx_pos_progress_mask, extra_pos_progress_mask], dim = 0)

                openx_len = len(openx_pos_video_array)
                extra_len = len(extra_pos_video_array)

                batch_triangular_mask = triangular_mask.repeat(openx_len + extra_len, 1, 1, 1).bool()

                
                video_embedding = video_encoder(video_array, batch_triangular_mask)
                text_array = text_encoder(text_array, text_mask)
                text_array = text_array.unsqueeze(1).repeat(1, video_embedding.size(1), 1)

                video_embedding = video_embedding.view(video_embedding.size(0) * video_embedding.size(1), -1)
                text_array = text_array.view(text_array.size(0) * text_array.size(1), -1)

                # dot product
                if args.norm_length:
                    video_embedding = F.normalize(video_embedding, p=2, dim=1)
                    text_array = F.normalize(text_array, p=2, dim=1)

                dot_product = torch.sum(video_embedding * text_array, dim = 1)
                dot_product = dot_product.view(-1, args.max_length)

                openx_pred = dot_product[:openx_len]
                extra_pred = dot_product[openx_len:]

                openx_target = progress[:openx_len]
                extra_target = progress[openx_len:]

                openx_loss = progress_loss_function(openx_pred[openx_pos_progress_mask], openx_target[openx_pos_progress_mask])
                extra_loss = progress_loss_function(extra_pred[extra_pos_progress_mask], extra_target[extra_pos_progress_mask])

                loss = (1 - args.extra_data_ratio) * openx_loss + args.extra_data_ratio * extra_loss

                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                wandb_log = {"openx_loss": openx_loss.item(), "extra_loss": extra_loss.item(), "loss": loss.item()}
                wandb_log["lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(wandb_log)

     


        with torch.no_grad():
            video_encoder.eval()
            text_encoder.eval()
            wandb_eval_log = {}




            if extra_eval_eval_pos_dataset is not None:
                extra_pos_progress_loss = eval_model_token(extra_eval_eval_pos_dataloader, 
                                                           video_encoder, 
                                                           text_encoder,
                                                           progress_loss_function, 
                                                           triangular_mask, 
                                                           args, 
                                                           pca_text_model = pca_text_model, 
                                                           pca_video_model = pca_video_model, 
                                                           neg_sample=False)
                wandb_eval_log["extra_eval/paired_progress_loss"] = extra_pos_progress_loss
            
            if extra_eval_eval_neg_dataset is not None:
                extra_neg_progress_loss = eval_model_token(extra_eval_eval_neg_dataloader, 
                                                           video_encoder, 
                                                           text_encoder,
                                                           progress_loss_function, 
                                                           triangular_mask, 
                                                           args, 
                                                           pca_text_model = pca_text_model, 
                                                           pca_video_model = pca_video_model, 
                                                           neg_sample=True)
                wandb_eval_log["extra_eval/unpaired_progress_loss"] = extra_neg_progress_loss

            if positive_eval_openx_dataset is not None:
                openx_pos_progress_loss = eval_model_token(openx_positive_eval_dataloader, 
                                                           video_encoder, 
                                                           text_encoder,
                                                           progress_loss_function, 
                                                           triangular_mask, 
                                                           args,
                                                           pca_text_model = pca_text_model, 
                                                           pca_video_model = pca_video_model, 
                                                           neg_sample=False)
          
                wandb_eval_log["openx_eval/paired_progress_loss"] = openx_pos_progress_loss
            
            if negative_eval_openx_dataset is not None:
                openx_neg_progress_loss = eval_model_token(openx_negative_eval_dataloader, 
                                                           video_encoder, 
                                                           text_encoder,
                                                           progress_loss_function,
                                                           triangular_mask,
                                                           args,
                                                           pca_text_model = pca_text_model,
                                                           pca_video_model = pca_video_model,
                                                           neg_sample=True)
                
                wandb_eval_log["openx_eval/unpaired_progress_loss"] = openx_neg_progress_loss

            wandb.log(wandb_eval_log)

            if epoch % 3 == 0:
                video_encoder.eval()
                text_encoder.eval()
                with torch.no_grad():
                    if args.extra_data_type == "metaworld":
                        plot_progress(h5_train_eval_file, 
                                      "train", 
                                      video_encoder, 
                                      text_encoder,
                                      args, 
                                      pca_text_model = pca_text_model, 
                                      pca_video_model = pca_video_model)
                        
                        plot_confusion_matrix(h5_file = h5_train_eval_file,
                                            set = "train",
                                            video_encoder = video_encoder,
                                            text_encoder= text_encoder,
                                            args = args,
                                            pca_text_model = pca_text_model,
                                            pca_video_model = pca_video_model)

                        plot_progress(h5_eval_file, 
                                      "eval", 
                                      video_encoder, 
                                      text_encoder,
                                      args, 
                                      pca_text_model = pca_text_model, 
                                      pca_video_model = pca_video_model)
                        plot_confusion_matrix(h5_file = h5_eval_file,
                                            set = "eval",
                                            video_encoder = video_encoder,
                                            text_encoder= text_encoder,
                                            args = args,
                                            pca_text_model = pca_text_model,
                                            pca_video_model = pca_video_model)               

                    else:
                        plot_progress(h5_train_eval_file, 
                                      "train", 
                                      video_encoder, 
                                      text_encoder,
                                      args, 
                                      pca_text_model = pca_text_model, 
                                      pca_video_model = pca_video_model)
                        plot_progress(h5_eval_file, 
                                      "eval", 
                                      video_encoder, 
                                      text_encoder,
                                      args, 
                                      pca_text_model = pca_text_model, 
                                      pca_video_model = pca_video_model)
                        plot_confusion_matrix(h5_file = h5_train_eval_file,
                                            set = "train",
                                            video_encoder = video_encoder,
                                            text_encoder= text_encoder,
                                            args = args, 
                                            pca_text_model = pca_text_model, 
                                            pca_video_model = pca_video_model)
                        plot_confusion_matrix(h5_file = h5_eval_file,
                                            set = "eval",
                                            video_encoder = video_encoder,
                                            text_encoder= text_encoder,
                                            args = args,
                                            pca_text_model = pca_text_model,
                                            pca_video_model = pca_video_model)





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
    argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/home/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    # argparser.add_argument('--h5_embedding_path', type=str, default='/mnt/ssd_a_4tb/jzhang96/openx_embeddings_full_uncompressed_with_langtable_processed.h5')
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld", "real_world"], default="real_world")
    argparser.add_argument('--batch_size', type=int, default=1024)
    argparser.add_argument('--catagorical_progress', action='store_true')
    argparser.add_argument('--two_step_training', action='store_true')
    argparser.add_argument('--epochs', type=int, default=10000)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=1)
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
    argparser.add_argument('--view', type=str, default="side", choices=["side", "top"])
    argparser.add_argument('--extra_data_ratio', type=float, default=0.02)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--norm_length', action='store_true')
    argparser.add_argument('--negative_mask_ratio', type=float, default=0.8)
    argparser.add_argument('--roll_percentage', type=float, default=1.0)

    args = argparser.parse_args()
    main(args)






    