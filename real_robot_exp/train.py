import torch
# from dataloader_liv_decoder_5_demo import video_collate_triangular_fn, LivVideoDecoderDataset5Frames
from dataset import LivRealVideoDataset, LivRealVideoEvalDataset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import argparse
import wandb
from tqdm import tqdm
import h5py
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from models import RewardPredictor, RewardTwoStepPredictor, RewardTwoStepNewPositionEmbeddingPredictor
from eval_confusion_matrix import plot_confusion_matrix
from eval_progress import plot_progress
from eval_raw_video_progress import real_video_plot
from utils import update_model, CosineWithMinLRScheduler
from torch.optim import Optimizer


os.environ["TOKENIZERS_PARALLELISM"] = "False"






def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "OpenXLIVLangTableOneLinearCosine"


    experiment_name += "_heads_" + str(args.attention_heads)

    if args.sample_neg:
        experiment_name += "_SampleNeg"
    if args.reverse_video:
        experiment_name += "_ReverseVideo"
    if args.catagorical_progress:
        experiment_name += "_CatProgress"
    if args.subsample_video:
        experiment_name += "_SubVideo"
        experiment_name += "_MaxLen" + str(args.max_length)
    if args.positional_encoding:
        experiment_name += "_PosEmb"
    if args.extra_data:
        experiment_name += "_ExtraData"
    if args.openx_data:
        experiment_name += "_OpenXData"
    if args.two_step_training:
        experiment_name += "_TwoStep"
    if args.cat_text:
        experiment_name += "_CatText"
    if args.layer_norm:
        experiment_name += "_LayerNorm"
    if args.first_frame_embedding:
        experiment_name += "_FirstFrameEmb"
    if args.cat_text_front:
        experiment_name += "_CatTextFront"
    if args.learner_parameter:
        experiment_name += "_LearnerPara"
    if args.cosine_scheduler:
        experiment_name += "_CosScheduler"
    
    experiment_name += "_DecoderNum_" + str(args.decoder_num)
    experiment_name += "_epochs_" + str(args.epochs)
    experiment_name += "_lr_" + str(args.lr)
    # experiment_name += "_1_demo"
    
    
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Jan24ndOpenXVideo",
        config=args,
        name=experiment_name,
    )



    # h5_file = h5py.File(args.h5_embedding_path, "r")
    h5_eval_file = h5py.File("jesse_collect_dataset_new.h5", "r")
    embedding_dim = 1024


    eval_dataset = None
    eval_dataloader = None
    if args.openx_data and args.extra_data:
        openx_dataset = LivRealVideoDataset(args, args.h5_embedding_path, split = False)
        extra_dataset = LivRealVideoDataset(args, "jesse_collect_dataset_new.h5", split = True)
        openx_dataloader = DataLoader(openx_dataset, batch_size=args.batch_size * 5, shuffle=True, num_workers=int(args.worker * 1.9), drop_last=True, pin_memory=True)
        extra_dataloader = DataLoader(extra_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
        batch_size = args.batch_size + args.batch_size * 5

        # positive_eval_dataset = LivRealVideoEvalDataset(args, 
        #                                                 "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5",
        #                                                 label = "positive")
        # negative_eval_dataset = LivRealVideoEvalDataset(args,
        #                                                 "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5",
        #                                                 label = "negative")
        
        positive_eval_dataset = LivRealVideoEvalDataset(args, 
                                                        "/mnt/ssd_a_4tb/jzhang96/openx_embeddings_test_dataset_progrssed.h5",
                                                        label = "positive")
        negative_eval_dataset = LivRealVideoEvalDataset(args,
                                                        "/mnt/ssd_a_4tb/jzhang96/openx_embeddings_test_dataset_progrssed.h5",
                                                        label = "negative")
        
        positive_eval_dataloader = DataLoader(positive_eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
        negative_eval_dataloader = DataLoader(negative_eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
        


    elif args.extra_data:
        dataset = LivRealVideoDataset(args, "jesse_collect_dataset_new.h5", split = True)
        extra_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
        batch_size = args.batch_size
    elif args.openx_data:
        dataset = LivRealVideoDataset(args, args.h5_embedding_path, split = False)
        openx_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
        batch_size = args.batch_size
    else:
        assert False, "No dataset specified"


    if args.two_step_training:
        classification_loss_function = CrossEntropyLoss()
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



    if args.catagorical_progress:
        if args.sample_neg:
            num_bins = args.catagorical_progress_bins + 1
        else:
            num_bins = args.catagorical_progress_bins
    else:
        num_bins = 1
    if args.two_step_training:
        if args.cat_text_front:
            self_attention_model = RewardTwoStepNewPositionEmbeddingPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
        else:
            self_attention_model = RewardTwoStepPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
    else:
        self_attention_model = RewardPredictor(embedding_dim, args = args, class_num=num_bins).to(device)

    print(self_attention_model)
    if args.cosine_scheduler:
        optimizer = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)], lr=args.lr)
        scheduler = CosineWithMinLRScheduler(optimizer, max_steps=120000, max_lr=args.lr, min_lr=1e-6)
    else:
        optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)
        scheduler = None

    if args.cat_text_front:
        triangular_mask = torch.tril(torch.ones(args.max_length + 1, args.max_length + 1)).to(device).unsqueeze(0).unsqueeze(0)
    else:
        triangular_mask = torch.tril(torch.ones(args.max_length, args.max_length)).to(device).unsqueeze(0).unsqueeze(0)

    batch_triangular_mask = triangular_mask.repeat(batch_size, 1, 1, 1).bool()


    # for epoch in range(args.epochs):
    #     self_attention_model.train()

    for epoch in range(args.epochs):
        self_attention_model.train()

        if args.openx_data and args.extra_data:
            for openx_data, extra_data in tqdm(zip(openx_dataloader, extra_dataloader), total = 100):
                openx_len = len(openx_data["video_array"])
                extra_len = len(extra_data["video_array"])

                video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
                text_array = torch.cat([openx_data["text_array"], extra_data["text_array"]], dim = 0).to(device).float().squeeze(1)

                progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)
                class_label = torch.cat([openx_data["class_label"], extra_data["class_label"]], dim = 0).to(device)

                wandb_log, self_attention_model = update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label,
                classification_loss_function, progress_loss_function, optimizer, openx_len = openx_len, extra_len = extra_len, scheduler = scheduler)
                wandb.log(wandb_log)
            
        elif args.openx_data:
            for openx_data in tqdm(openx_dataloader):
                video_array = openx_data["video_array"].to(device).float()
                text_array = openx_data["text_array"].to(device).float().squeeze(1)
                progress = openx_data["progress"].to(device)
                class_label = openx_data["class_label"].to(device)

                wandb_log, self_attention_model = update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label,
                classification_loss_function, progress_loss_function, optimizer)
                wandb.log(wandb_log)

        elif args.extra_data:
            for extra_data in tqdm(extra_dataloader):
                video_array = extra_data["video_array"].to(device).float()
                text_array = extra_data["text_array"].to(device).float().squeeze(1)
                progress = extra_data["progress"].to(device)
                class_label = extra_data["class_label"].to(device)

                wandb_log, self_attention_model = update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label,
                classification_loss_function, progress_loss_function, optimizer)
                wandb.log(wandb_log)

        












            

        if epoch % 20 == 0:
            self_attention_model.eval()
            with torch.no_grad():
        #     save_path = os.path.join("/scr/jzhang96/roboclip_v2_decoder_models_fix_3rd", experiment_name)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     save_dict = {
        #         "model": self_attention_model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch,
        #         "args": args
        #     }
        #     torch.save(save_dict, os.path.join(save_path, f"model_{epoch}.pth"))


            # real_video_plot(self_attention_model)
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
                
                if positive_eval_dataloader is not None:
                    correct_num = 0
                    total_num = 0
                    total_loss = 0
                    wrong_num = 0
                    for eval_data in tqdm(positive_eval_dataloader):
                        
                        video_array = eval_data["video_array"].to(device).float()
                        text_array = eval_data["text_array"].to(device).float().squeeze(1)
                        progress = eval_data["progress"].to(device)
                        class_label = eval_data["class_label"].to(device)
                        eval_batch_size, seq_len, _ = video_array.size()
                        eval_triangular_mask = triangular_mask.repeat(eval_batch_size, 1, 1, 1).bool()
                        progress_output, class_output = self_attention_model(video_array, eval_triangular_mask, text_array, mask = None)
                        
                        

                        class_label = class_label.view(eval_batch_size * seq_len)
                        progress = progress.view(eval_batch_size * seq_len, -1)
                        
                        class_output = class_output.view(eval_batch_size * seq_len, -1)
                        progress_output = progress_output.view(eval_batch_size * seq_len, -1)

                        none_zero_class = class_label != 0
                        progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class])
                        class_predict_label = torch.argmax(class_output, dim=1)
                        # class_accuracy = torch.sum(class_predict_label == class_label).item() / len(class_predict_label)
                        correct_num += torch.sum(class_predict_label == class_label).item()
                        total_num += len(class_predict_label)
                        total_loss += progress_loss.item()
                    class_accuracy = correct_num / total_num
                    progress = total_loss / total_num

                    wandb_eval_log = {
                        "openx_eval/progress_loss": progress_loss,
                        "openx_eval/correct_class_accuracy": class_accuracy
                    }


                    for eval_data in tqdm(negative_eval_dataloader):
                        
                        video_array = eval_data["video_array"].to(device).float()
                        text_array = eval_data["text_array"].to(device).float().squeeze(1)
                        progress = eval_data["progress"].to(device)
                        class_label = eval_data["class_label"].to(device)
                        eval_batch_size, seq_len, _ = video_array.size()
                        eval_triangular_mask = triangular_mask.repeat(eval_batch_size, 1, 1, 1).bool()
                        _, class_output = self_attention_model(video_array, eval_triangular_mask, text_array, mask = None)
                        
                        class_label = class_label.view(eval_batch_size * seq_len)
                        progress = progress.view(eval_batch_size * seq_len, -1)
                        
                        class_output = class_output.view(eval_batch_size * seq_len, -1)

                        none_zero_class = class_label != 0
                        class_predict_label = torch.argmax(class_output, dim=1)
                        wrong_num += torch.sum(class_predict_label == class_label).item()
                        # # correct_num += torch.sum(class_predict_label == class_label).item()
                        # wrong_num += torch.sum(class_predict_label != class_label).item()
                        # total_num += len(class_predict_label)
                        # total_loss += progress_loss.item()
                    wrong_class_accuracy = wrong_num / total_num

                    wandb_eval_log["openx_eval/wrong_class_accuracy"] = wrong_class_accuracy


                    wandb.log(wandb_eval_log)




            


        # if epoch % 50 == 49:
        #     self_attention_model.eval()
        #     plot_videos_class("liv", self_attention_model, args)



            






if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable35k_processed.h5')
    argparser.add_argument('--h5_embedding_path', type=str, default='/mnt/ssd_a_4tb/jzhang96/openx_embeddings_full_uncompressed_with_langtable35k_processed.h5')
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=200)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--sample_neg', action='store_true')
    argparser.add_argument('--reverse_video', action='store_true')
    argparser.add_argument('--normalize_embedding', action='store_true')    
    argparser.add_argument('--catagorical_progress', action='store_true')
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--catagorical_progress_bins', type=int, default=5)
    argparser.add_argument('--max_length', type=int, default=32)
    argparser.add_argument('--positional_encoding', action='store_true')
    argparser.add_argument('--extra_data', action='store_true')
    argparser.add_argument('--openx_data', action='store_true')
    argparser.add_argument('--layer_norm', action='store_true')
    argparser.add_argument('--two_step_training', action='store_true')
    argparser.add_argument('--cat_text', action='store_true')
    argparser.add_argument('--decoder_num', type=int, default=1)
    argparser.add_argument('--first_frame_embedding', action='store_true')
    argparser.add_argument('--cat_text_front', action='store_true')
    argparser.add_argument('--learner_parameter', action='store_true')
    argparser.add_argument('--cosine_scheduler', action='store_true')
    args = argparser.parse_args()
    main(args)






    