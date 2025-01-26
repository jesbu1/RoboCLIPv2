import torch
# from dataloader_liv_decoder_5_demo import video_collate_triangular_fn, LivVideoDecoderDataset5Frames
from dataset import LivRealVideoTextTokenDataset, VideoTokenCollateTriangularFn
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
from models import  RewardTwoStepLangTokenPositionEmbeddingPredictor
from eval_confusion_matrix import plot_confusion_matrix_token
from eval_progress import plot_progress_token
from torch.optim import Optimizer
import math

os.environ["TOKENIZERS_PARALLELISM"] = "False"

class CosineWithMinLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            # Cosine decay for the first max_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos_decay for _ in self.base_lrs]
        else:
            # Keep the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]




def TokenPositionEmbedding(max_seq_len, embed_dim):
    position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: [max_seq_len, 1]
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
    pe = torch.zeros(max_seq_len, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    pe /= 10 # reduce the scale of positional encoding otherwise it will dominate the input embeddings
    return pe.unsqueeze(0)






def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "OpenXLIVCatLangTokenAddPosEmb"


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
    if args.clip_grad:
        experiment_name += "_ClipGrad"

    experiment_name += "_DecoderNum_" + str(args.decoder_num)
    experiment_name += "_epochs_" + str(args.epochs)
    experiment_name += "_lr_" + str(args.lr)

    
    # experiment_name += "_1_demo"
    
    
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="Jan25ndOpenXVideoTokenLog",
        config=args,
        name=experiment_name,
    )



    # h5_file = h5py.File(args.h5_embedding_path, "r")
    h5_eval_file = h5py.File("jesse_collect_dataset_new_token.h5", "r")
    embedding_dim = 1024
    max_seq_len = 45


    eval_dataset = None
    eval_dataloader = None
    if args.openx_data and args.extra_data:
        openx_dataset = LivRealVideoTextTokenDataset(args, args.h5_embedding_path, split = False)
        extra_dataset = LivRealVideoTextTokenDataset(args, "jesse_collect_dataset_new_token.h5", split = True)
        openx_dataloader = DataLoader(openx_dataset, 
                                      batch_size=args.batch_size * 5, 
                                      shuffle=True, 
                                      num_workers=args.worker * 2, 
                                      drop_last=True, 
                                      pin_memory=True, 
                                      collate_fn = VideoTokenCollateTriangularFn)
        extra_dataloader = DataLoader(extra_dataset, 
                                      batch_size=args.batch_size, 
                                      shuffle=True, 
                                      num_workers=args.worker, 
                                      drop_last=True, 
                                      pin_memory=True, 
                                      collate_fn = VideoTokenCollateTriangularFn)
        batch_size = args.batch_size + args.batch_size * 5

    #     positive_eval_dataset = LivRealVideoEvalDataset(args, 
    #                                                     "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5",
    #                                                     label = "positive")
    #     negative_eval_dataset = LivRealVideoEvalDataset(args,
    #                                                     "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_test_dataset_progrssed.h5",
    #                                                     label = "negative")
        
        # positive_eval_dataloader = DataLoader(positive_eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
        # negative_eval_dataloader = DataLoader(negative_eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
        




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
            self_attention_model = RewardTwoStepLangTokenPositionEmbeddingPredictor(embedding_dim, args = args, class_num=num_bins).to(device)
        else:
            assert False, "Not implemented"
    else:
        assert False, "Not implemented"

    # print(self_attention_model)

    


    if args.learner_parameter:
        input_dim = 1024
        text_learner_parameter = torch.nn.Parameter(torch.randn(1, input_dim, device = device))
        video_learner_parameter = torch.nn.Parameter(torch.randn(1, input_dim, device = device))
        optimizer = torch.optim.Adam(list(self_attention_model.parameters()) + [text_learner_parameter, video_learner_parameter], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(self_attention_model.parameters(), lr=args.lr)

    if args.cosine_scheduler:
        scheduler = CosineWithMinLRScheduler(optimizer, max_steps = 120000, max_lr = args.lr, min_lr = 1e-6)
    # batch_triangular_mask = triangular_mask.repeat(batch_size, 1, 1, 1).bool()


    text_position_embedding = TokenPositionEmbedding(max_seq_len, 1024).to(device)


    for epoch in range(args.epochs):
        self_attention_model.train()

        if args.openx_data and args.extra_data:
            for openx_data, extra_data in tqdm(zip(openx_dataloader, extra_dataloader), total = 100):
                max_len = max(openx_data["text_output"].shape[1], extra_data["text_output"].shape[1])
                openx_len = openx_data["text_output"].shape[1]
                extra_len = extra_data["text_output"].shape[1]

                # add text position embedding to text_output
                openx_data["text_output"] = openx_data["text_output"].to(device) + text_position_embedding[:, :openx_len]
                extra_data["text_output"] = extra_data["text_output"].to(device) + text_position_embedding[:, :extra_len]

                openx_batch_size, openx_text_seq_len, _ = openx_data["text_output"].size()
                extra_batch_size, extra_text_seq_len, _ = extra_data["text_output"].size()
                
                # add learnerable parameters to text and video output
                openx_text_output = openx_data["text_output"].to(device).float().view(-1, 1024)
                extra_text_output = extra_data["text_output"].to(device).float().view(-1, 1024)
                openx_video_output = openx_data["video_output"].to(device).float().view(-1, 1024)
                extra_video_output = extra_data["video_output"].to(device).float().view(-1, 1024)

                openx_text_output += text_learner_parameter
                extra_text_output += text_learner_parameter

                openx_video_output += video_learner_parameter
                extra_video_output += video_learner_parameter

                openx_text_output = openx_text_output.view(openx_batch_size, openx_text_seq_len, 1024)
                extra_text_output = extra_text_output.view(extra_batch_size, extra_text_seq_len, 1024)
                openx_video_output = openx_video_output.view(openx_batch_size, args.max_length, 1024)
                extra_video_output = extra_video_output.view(extra_batch_size, args.max_length, 1024)

                openx_mask = openx_data["mask_output"].to(device)
                extra_mask = extra_data["mask_output"].to(device)

                # padding text to same length
                if openx_len < max_len:
                    openx_text_output = F.pad(openx_text_output, (0, 0, 0, max_len - openx_len))
                    openx_mask = F.pad(openx_mask, (0, max_len - openx_len))
                elif extra_len < max_len:
                    extra_text_output = F.pad(extra_text_output, (0, 0, 0, max_len - extra_len))
                    extra_mask = F.pad(extra_mask, (0, max_len - extra_len))

                total_video_output = torch.cat([extra_video_output, openx_video_output], dim = 0)
                total_text_output = torch.cat([extra_text_output, openx_text_output], dim = 0)
                total_mask = torch.cat([extra_mask, openx_mask], dim = 0)

                feature_input_list = list()
                video_mask_list = list()
                text_start_list = list()
                for i in range(len(total_video_output)):
                    text = total_text_output[i]
                    video = total_video_output[i]
                    mask = total_mask[i]
                    front_text = text[mask==1]
                    back_text = text[mask==0]
                    feature_input = torch.cat([front_text, video, back_text], dim = 0)
                    video_mask = torch.zeros(len(feature_input), dtype = torch.bool)
                    video_mask[len(front_text):len(front_text) + len(video)] = True
                    video_mask_list.append(video_mask)
                    feature_input_list.append(feature_input)
                    text_start_list.append(len(front_text))
                feature_input = torch.stack(feature_input_list, dim = 0)
                video_mask = torch.stack(video_mask_list, dim = 0)
                video_start = torch.tensor(text_start_list, device = device)
                triangular_mask = torch.tril(torch.ones(feature_input.shape[1], feature_input.shape[1])).to(device).unsqueeze(0).unsqueeze(0)
                batch_triangular_mask = triangular_mask.repeat(openx_batch_size + extra_batch_size, 1, 1, 1).bool()
                progress_output, class_output = self_attention_model(feature_input, batch_triangular_mask, video_start, mask = video_mask)

                extra_progress_output = progress_output[:extra_batch_size].view(-1, 1)
                openx_progress_output = progress_output[extra_batch_size:].view(-1, 1)

                extra_two_step_pred = class_output[:extra_batch_size].view(-1, 2)
                openx_two_step_pred = class_output[extra_batch_size:].view(-1, 2)


                openx_progress_label = openx_data["progress_output"].to(device).float().view(-1, 1)
                extra_progress_label = extra_data["progress_output"].to(device).float().view(-1, 1)

                openx_class_label = openx_data["class_label_output"].to(device).long().view(-1)
                extra_class_label = extra_data["class_label_output"].to(device).long().view(-1)

                extra_progress_loss = progress_loss_function(extra_progress_output, extra_progress_label)
                openx_progress_loss = progress_loss_function(openx_progress_output, openx_progress_label)

                extra_class_loss = classification_loss_function(extra_two_step_pred, extra_class_label)
                openx_class_loss = classification_loss_function(openx_two_step_pred, openx_class_label) 

                
                total_loss = extra_progress_loss + openx_progress_loss + extra_class_loss + openx_class_loss

                openx_accuracy = torch.sum(torch.argmax(openx_two_step_pred, dim = 1) == openx_class_label).item() / len(openx_class_label)
                openx_accuracy_one = torch.sum(torch.argmax(openx_two_step_pred[openx_class_label==1], dim = 1) == openx_class_label[openx_class_label==1]).item() / len(openx_class_label[openx_class_label==1])
                openx_accuracy_zero = torch.sum(torch.argmax(openx_two_step_pred[openx_class_label==0], dim = 1) == openx_class_label[openx_class_label==0]).item() / len(openx_class_label[openx_class_label==0])

                extra_accuracy = torch.sum(torch.argmax(extra_two_step_pred, dim = 1) == extra_class_label).item() / len(extra_class_label)
                extra_accuracy_one = torch.sum(torch.argmax(extra_two_step_pred[extra_class_label==1], dim = 1) == extra_class_label[extra_class_label==1]).item() / len(extra_class_label[extra_class_label==1])
                extra_accuracy_zero = torch.sum(torch.argmax(extra_two_step_pred[extra_class_label==0], dim = 1) == extra_class_label[extra_class_label==0]).item() / len(extra_class_label[extra_class_label==0])

                wandb_log = {
                    "loss/openx_progress_loss": openx_progress_loss.item(),
                    "loss/extra_progress_loss": extra_progress_loss.item(),
                    "loss/openx_class_loss": openx_class_loss.item(),
                    "loss/extra_class_loss": extra_class_loss.item(),
                    "loss/total_loss": total_loss.item(),
                    "accuracy/openx_accuracy": openx_accuracy,
                    "accuracy/openx_accuracy_one": openx_accuracy_one,
                    "accuracy/openx_accuracy_zero": openx_accuracy_zero,
                    "accuracy/extra_accuracy": extra_accuracy,
                    "accuracy/extra_accuracy_one": extra_accuracy_one,
                    "accuracy/extra_accuracy_zero": extra_accuracy_zero
                }


                optimizer.zero_grad()
                total_loss.backward()
                if args.clip_grad:
                    if args.learner_parameter:
                        torch.nn.utils.clip_grad_norm_(list(self_attention_model.parameters()) + [text_learner_parameter, video_learner_parameter], 1)
                    else:
                        torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), 1)

                optimizer.step()
                wandb_log["lr"] = optimizer.param_groups[0]["lr"]
                if args.cosine_scheduler:
                    scheduler.step()

                wandb.log(wandb_log)


        if epoch % 3 == 0:
            self_attention_model.eval()
            with torch.no_grad():
        # #     save_path = os.path.join("/scr/jzhang96/roboclip_v2_decoder_models_fix_3rd", experiment_name)
        # #     if not os.path.exists(save_path):
        # #         os.makedirs(save_path)
        # #     save_dict = {
        # #         "model": self_attention_model.state_dict(),
        # #         "optimizer": optimizer.state_dict(),
        # #         "epoch": epoch,
        # #         "args": args
        # #     }
        # #     torch.save(save_dict, os.path.join(save_path, f"model_{epoch}.pth"))


        #     # real_video_plot(self_attention_model)
                plot_progress_token(h5_eval_file, "train", self_attention_model, args)
                plot_progress_token(h5_eval_file, "eval", self_attention_model, args)
                plot_confusion_matrix_token(h5_file = h5_eval_file, 
                                        set = "train", 
                                        self_attention_model = self_attention_model, 
                                        args = args)
                plot_confusion_matrix_token(h5_file = h5_eval_file, 
                                        set = "eval", 
                                        self_attention_model = self_attention_model, 
                                        args = args)

                plot_confusion_matrix_token(h5_file = h5_eval_file, 
                                        set = "all", 
                                        self_attention_model = self_attention_model, 
                                        args = args)
                
        #         if positive_eval_dataloader is not None:
        #             correct_num = 0
        #             total_num = 0
        #             total_loss = 0
        #             wrong_num = 0
        #             for eval_data in tqdm(positive_eval_dataloader):
                        
        #                 video_array = eval_data["video_array"].to(device).float()
        #                 text_array = eval_data["text_array"].to(device).float().squeeze(1)
        #                 progress = eval_data["progress"].to(device)
        #                 class_label = eval_data["class_label"].to(device)
        #                 eval_batch_size, seq_len, _ = video_array.size()
        #                 eval_triangular_mask = triangular_mask.repeat(eval_batch_size, 1, 1, 1).bool()
        #                 progress_output, class_output = self_attention_model(video_array, eval_triangular_mask, text_array, mask = None)
                        
                        

        #                 class_label = class_label.view(eval_batch_size * seq_len)
        #                 progress = progress.view(eval_batch_size * seq_len, -1)
                        
        #                 class_output = class_output.view(eval_batch_size * seq_len, -1)
        #                 progress_output = progress_output.view(eval_batch_size * seq_len, -1)

        #                 none_zero_class = class_label != 0
        #                 progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class])
        #                 class_predict_label = torch.argmax(class_output, dim=1)
        #                 # class_accuracy = torch.sum(class_predict_label == class_label).item() / len(class_predict_label)
        #                 correct_num += torch.sum(class_predict_label == class_label).item()
        #                 total_num += len(class_predict_label)
        #                 total_loss += progress_loss.item()
        #             class_accuracy = correct_num / total_num
        #             progress = total_loss / total_num

        #             wandb_eval_log = {
        #                 "openx_eval/progress_loss": progress_loss,
        #                 "openx_eval/correct_class_accuracy": class_accuracy
        #             }


        #             for eval_data in tqdm(negative_eval_dataloader):
                        
        #                 video_array = eval_data["video_array"].to(device).float()
        #                 text_array = eval_data["text_array"].to(device).float().squeeze(1)
        #                 progress = eval_data["progress"].to(device)
        #                 class_label = eval_data["class_label"].to(device)
        #                 eval_batch_size, seq_len, _ = video_array.size()
        #                 eval_triangular_mask = triangular_mask.repeat(eval_batch_size, 1, 1, 1).bool()
        #                 _, class_output = self_attention_model(video_array, eval_triangular_mask, text_array, mask = None)
                        
        #                 class_label = class_label.view(eval_batch_size * seq_len)
        #                 progress = progress.view(eval_batch_size * seq_len, -1)
                        
        #                 class_output = class_output.view(eval_batch_size * seq_len, -1)

        #                 none_zero_class = class_label != 0
        #                 class_predict_label = torch.argmax(class_output, dim=1)
        #                 wrong_num += torch.sum(class_predict_label == class_label).item()
        #                 # # correct_num += torch.sum(class_predict_label == class_label).item()
        #                 # wrong_num += torch.sum(class_predict_label != class_label).item()
        #                 # total_num += len(class_predict_label)
        #                 # total_loss += progress_loss.item()
        #             wrong_class_accuracy = wrong_num / total_num

        #             wandb_eval_log["openx_eval/wrong_class_accuracy"] = wrong_class_accuracy


        #             wandb.log(wandb_eval_log)




            


        # if epoch % 50 == 49:
        #     self_attention_model.eval()
        #     plot_videos_class("liv", self_attention_model, args)



            







if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable35k_processed.h5')
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
    argparser.add_argument('--clip_grad', action='store_true')
    args = argparser.parse_args()
    main(args)






    