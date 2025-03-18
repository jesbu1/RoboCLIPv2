import torch
from PIL import Image
from dataloader_liv_decoder_5_demo import video_collate_triangular_fn, LivVideoDecoderDataset5Frames
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
import wandb
from tqdm import tqdm
import h5py
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss 
import os
from decoder_only_models import RewardPredictor, RewardTwoStepPredictor
from eval_utils_decoder_5_demos import plot_progress_class, plot_videos_class
from confusion_matrix_decoder_5_demos import plot_confusion_matrix_pca_class



os.environ["TOKENIZERS_PARALLELISM"] = "False"

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "RegressionFixTransformer5Demos_" + args.model_name 

    if args.sample_neg:
        experiment_name += "_sample_neg"


    experiment_name += "_heads_" + str(args.attention_heads)

    if args.sample_neg:
        experiment_name += "_SampleNeg"
    if args.reverse_video:
        experiment_name += "_ReverseVideo"
    if args.normalize_embedding:
        experiment_name += "_norm"
    if args.catagorical_progress:
        experiment_name += "_CatProgress"
    if args.subsample_video:
        experiment_name += "_SubsampleVideo"
        experiment_name += "_MaxLen" + str(args.max_length)
    if args.cat_embedding:
        experiment_name += "_CatEmbedding"
    if args.enlarge_embedding_space:
        experiment_name += "_EnlargeEmbedding"
    if args.fully_reverse_data:
        experiment_name += "_FullyReverse"
    if args.two_step_training:
        experiment_name += "_TwoStep"
    if args.positional_encoding:
        experiment_name += "_PositionalEncoding"

    experiment_name += "_500"
    # experiment_name += "_1_demo"
    
    
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Regression_final_20task_Debug_2nd",
        config=args,
        name=experiment_name,
    )



    h5_file = h5py.File(args.h5_embedding_path, "r")
    h5_eval_file = h5py.File("metaworld_embedding_1_demo_dataset_v2.h5", "r")
    embedding_dim = 1024




    dataset = LivVideoDecoderDataset5Frames(args, h5_file)


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, collate_fn=video_collate_triangular_fn)

    if args.two_step_training:
        classification_loss_function = CrossEntropyLoss()
        if args.catagorical_progress:
            progress_loss_function = CrossEntropyLoss()
        else:
            progress_loss_function = mse_loss        
    elif args.catagorical_progress:
        progress_loss_function = CrossEntropyLoss()
    else:
        progress_loss_function = mse_loss


    if args.two_step_training:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
        else:
            num_bins = 1
        self_attention_model = RewardTwoStepPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
    else:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
        else:
            num_bins = 1
        self_attention_model = RewardPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
    print(self_attention_model)
    optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        self_attention_model.train()

        for i, data in enumerate(tqdm(dataloader)):
            video_array = data["video_array"].to(device).float()
            text_array = data["text_array"].to(device).float()

            # if not args.subsample_video:
            triangular_mask = data["triangular_mask"].to(device).float()
            mask = data["mask"].to(device).bool()
            progress_output, class_output = self_attention_model(video_array, triangular_mask.unsqueeze(1), text_array, mask)
                
            # else:
            #     progress_output, class_output = self_attention_model(video_array, None, text_array)
            if args.catagorical_progress:
                progress = data["progress"].to(device).long()
            else:
                progress = data["progress"].to(device).float().unsqueeze(2)

            if args.two_step_training:
                batch_size, seq_len, _ = video_array.size()
                mask = mask.view(batch_size * seq_len)
                class_label = data["class_label"].to(device).long()
                class_label = class_label.view(batch_size * seq_len, -1)[mask].squeeze(1)
                progress = progress.view(batch_size * seq_len, -1)[mask]
                class_loss = classification_loss_function(class_output, class_label)

                if args.catagorical_progress:
                    progress = progress.squeeze(1)
                none_zero_class = class_label != 0
                progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class])
                loss = class_loss + progress_loss
                class_predict_label = torch.argmax(class_output, dim=1)
                class_accuracy = torch.sum(class_predict_label == class_label).item() / len(class_predict_label)
                wandb_log = {
                    "total_loss": loss.item(),
                    "class_loss": class_loss.item(),
                    "progress_loss": progress_loss.item(),
                    "class_accuracy": class_accuracy
                }
                
                if args.catagorical_progress:
                    predict_label = torch.argmax(progress_output, dim=1)
                    progress_accuracy = torch.sum(predict_label == progress).item() / len(predict_label)
                    wandb_log["progress_accuracy"] = progress_accuracy


            else:
                batch_size, seq_len, _ = video_array.size()
                mask = mask.view(batch_size * seq_len)
                progress = progress.view(batch_size * seq_len, -1)
                if args.catagorical_progress:
                    progress = progress.squeeze(1)

                loss = progress_loss_function(progress_output, progress[mask])
                wandb_log = {
                    "progress_loss": loss.item(),
                }
                
                if args.catagorical_progress:
                    predict_label = torch.argmax(progress_output, dim=1)
                    accuracy = torch.sum(predict_label == progress[mask]) / len(predict_label)
                    wandb_log["progress_accuracy"] = accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(wandb_log)

        if epoch % 20 == 19:
            self_attention_model.eval()

            save_path = os.path.join("roboclip_v2_decoder_models_fix_3rd", experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_dict = {
                "model": self_attention_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args
            }
            torch.save(save_dict, os.path.join(save_path, f"model_{epoch}.pth"))


            # if args.catagorical_progress:
                
            plot_progress_class(h5_eval_file, "train", self_attention_model, args)
            plot_progress_class(h5_eval_file, "eval", self_attention_model, args)
            plot_confusion_matrix_pca_class(h5_file = h5_eval_file, 
                                    set = "train", 
                                    self_attention_model = self_attention_model, 
                                    args = args)
            plot_confusion_matrix_pca_class(h5_file = h5_eval_file,
                                            set = "eval",
                                            self_attention_model = self_attention_model,
                                            args = args)
            


        # if epoch % 50 == 49:
        #     self_attention_model.eval()
        #     plot_videos_class(args.model_name, self_attention_model, args)



            







if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_25_for_clip_liv.h5')
    argparser.add_argument('--h5_embedding_path', type=str, default='metaworld_embedding_5_demo_dataset_v2.h5')
    argparser.add_argument('--model_name', type=str, default='liv', choices=['clip', 'liv'])
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=200)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=4)
    argparser.add_argument('--attention_heads', type=int, default=4)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--sample_neg', action='store_true')
    argparser.add_argument('--enlarge_embedding_space', action='store_true')
    argparser.add_argument('--reverse_video', action='store_true')
    argparser.add_argument('--normalize_embedding', action='store_true')    
    argparser.add_argument('--catagorical_progress', action='store_true')
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--catagorical_progress_bins', type=int, default=5)
    argparser.add_argument('--max_length', type=int, default=32)
    argparser.add_argument('--cat_embedding', action='store_true')
    argparser.add_argument('--fully_reverse_data', action='store_true')
    argparser.add_argument('--two_step_training', action='store_true')
    argparser.add_argument('--layer_norm', action='store_true')
    argparser.add_argument('--positional_encoding', action='store_true')
    args = argparser.parse_args()
    main(args)






    