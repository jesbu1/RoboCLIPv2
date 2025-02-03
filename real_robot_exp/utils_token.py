import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim.optimizer import Optimizer








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

def get_cosine_positional_encoding(max_seq_len, embed_dim):
    """
    Generate a static positional encoding matrix using sine and cosine functions.
    """
    position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: [max_seq_len, 1]
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
    pe = torch.zeros(max_seq_len, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    pe /= 10 # reduce the scale of positional encoding otherwise it will dominate the input embeddings
    return pe.unsqueeze(0)



def update_model(args, total_input, mask, batch_triangular_mask, self_attention_model, progress, class_label, 
                classification_loss_function, progress_loss_function, optimizer,  openx_len, extra_len, scaler):

    progress_output, class_output = self_attention_model(total_input, batch_triangular_mask, mask = mask)

    if args.catagorical_progress:
        progress = progress.long()
    else:
        progress = progress.float().unsqueeze(2)

    if args.two_step_training:
        # batch_size, seq_len, _ = video_array.size()
        class_label = class_label.float()

        if openx_len is not None:
            openx_class_label = class_label[:openx_len].view(openx_len * args.max_length)
            extra_class_label = class_label[openx_len:].view(extra_len * args.max_length)

            openx_progress_label = progress[:openx_len].view(openx_len * args.max_length)
            extra_progress_label = progress[openx_len:].view(extra_len * args.max_length)

            pred_openx_class = class_output[:openx_len].view(openx_len * args.max_length, -1).squeeze(1)
            pred_extra_class = class_output[openx_len:].view(extra_len * args.max_length, -1).squeeze(1)
            openx_pred_progress = progress_output[:openx_len].view(openx_len * args.max_length, -1).squeeze(1)
            extra_pred_progress = progress_output[openx_len:].view(extra_len * args.max_length, -1).squeeze(1)

                
            openx_class_loss = classification_loss_function(pred_openx_class, openx_class_label)
            extra_class_loss = classification_loss_function(pred_extra_class, extra_class_label)

            class_loss = openx_class_loss + extra_class_loss

            openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
            extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)

            progress_loss = openx_progress_loss + extra_progress_loss

            loss = class_loss + progress_loss

            openx_none_zero_class = openx_class_label != 0
            extra_none_zero_class = extra_class_label != 0

            openx_progress_label_true = openx_progress_label[openx_none_zero_class]
            extra_progress_label_true = extra_progress_label[extra_none_zero_class]
            openx_pred_progress_true = openx_pred_progress[openx_none_zero_class]
            extra_pred_progress_true = extra_pred_progress[extra_none_zero_class]

            openx_zero_class = openx_class_label == 0
            extra_zero_class = extra_class_label == 0
            openx_progress_label_false = openx_progress_label[openx_zero_class]
            extra_progress_label_false = extra_progress_label[extra_zero_class]
            openx_pred_progress_false = openx_pred_progress[openx_zero_class]
            extra_pred_progress_false = extra_pred_progress[extra_zero_class]

            if args.catagorical_progress:
                openx_predict_label_true = torch.argmax(openx_pred_progress_true, dim=1)
                extra_predict_label_true = torch.argmax(extra_pred_progress_true, dim=1)
                openx_predict_label_false = torch.argmax(openx_pred_progress_false, dim=1)
                extra_predict_label_false = torch.argmax(extra_pred_progress_false, dim=1)
                openx_progress_accuracy_true = torch.sum(openx_predict_label_true == openx_progress_label_true).item() / len(openx_predict_label_true)
                extra_progress_accuracy_true = torch.sum(extra_predict_label_true == extra_progress_label_true).item() / len(extra_predict_label_true)
                openx_progress_accuracy_false = torch.sum(openx_predict_label_false == openx_progress_label_false).item() / len(openx_predict_label_false)
                extra_progress_accuracy_false = torch.sum(extra_predict_label_false == extra_progress_label_false).item() / len(extra_predict_label_false)
                openx_progress_accuracy = torch.sum(openx_predict_label_true == openx_progress_label_true).item() / len(openx_predict_label_true)
                extra_progress_accuracy = torch.sum(extra_predict_label_true == extra_progress_label_true).item() / len(extra_predict_label_true)

            
            pred_openx_class_label = pred_openx_class.round()
            pred_extra_class_label = pred_extra_class.round()

            openx_class_accuracy = torch.sum(pred_openx_class_label == openx_class_label).item() / len(pred_openx_class_label)
            extra_class_accuracy = torch.sum(pred_extra_class_label == extra_class_label).item() / len(pred_extra_class_label)

            wandb_log = {
                "loss/total_loss": loss.item(),
                "loss/class_loss": class_loss.item(),
                "loss/progress_loss": progress_loss.item(),
                "loss/openx_class_loss": openx_class_loss.item(),
                "loss/extra_class_loss": extra_class_loss.item(),
                "openx/openx_class_loss": openx_class_loss.item(),
                "extra/extra_class_loss": extra_class_loss.item(),

                "accuracy/openx_two_step_accuracy": openx_class_accuracy,
                "accuracy/extra_two_step_accuracy": extra_class_accuracy, 
                "openx/openx_two_step_accuracy": openx_class_accuracy,
                "extra/extra_two_step_accuracy": extra_class_accuracy,
               
                "loss/openx_progress_loss": openx_progress_loss.item(),
                "loss/extra_progress_loss": extra_progress_loss.item(),
                "openx/openx_progress_loss": openx_progress_loss.item(),
                "extra/extra_progress_loss": extra_progress_loss.item(),

                "train_accuracy/openx_true_data_progress_accuracy": openx_progress_accuracy_true,
                "train_accuracy/extra_true_data_progress_accuracy": extra_progress_accuracy_true,
                "train_accuracy/openx_wrong_data_progress_accuracy": openx_progress_accuracy_false,
                "train_accuracy/extra_wrong_data_progress_accuracy": extra_progress_accuracy_false,

                "openx/openx_true_data_progress_accuracy": openx_progress_accuracy_true,
                "extra/extra_true_data_progress_accuracy": extra_progress_accuracy_true,
                "openx/openx_wrong_data_progress_accuracy": openx_progress_accuracy_false,
                "extra/extra_wrong_data_progress_accuracy": extra_progress_accuracy_false,

                "train_accuracy/openx_progress_accuracy": openx_progress_accuracy,
                "train_accuracy/extra_progress_accuracy": extra_progress_accuracy,
                "openx/openx_progress_accuracy": openx_progress_accuracy,
                "extra/extra_progress_accuracy": extra_progress_accuracy,
            }



        #     else:
            
        #         class_label = class_label.view(batch_size * seq_len)
        #         progress = progress.view(batch_size * seq_len, -1)

        #         class_output = class_output.view(batch_size * seq_len, -1)
        #         progress_output = progress_output.view(batch_size * seq_len, -1)
        #         class_loss = classification_loss_function(class_output, class_label)

        #         none_zero_class = class_label != 0
        #         progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class])
        #         loss = class_loss + progress_loss
        #         class_predict_label = torch.argmax(class_output, dim=1)
        #         class_accuracy = torch.sum(class_predict_label == class_label).item() / len(class_predict_label)

        #         wandb_log = {
        #             "total_loss": loss.item(),
        #             "class_loss": class_loss.item(),
        #             "progress_loss": progress_loss.item(),
        #             "class_accuracy": class_accuracy
        #         }
                
        #     if args.catagorical_progress:
        #         predict_label = torch.argmax(progress_output, dim=1)
        #         progress_accuracy = torch.sum(predict_label == progress).item() / len(predict_label)
        #         wandb_log["progress_accuracy"] = progress_accuracy

        # else:
        #     batch_size, seq_len, _ = video_array.size()
        #     progress = progress.view(batch_size * seq_len, -1)
        #     if args.catagorical_progress:
        #         progress = progress.squeeze(1)
        #     loss = progress_loss_function(progress_output, progress)
        #     wandb_log = {
        #         "progress_loss": loss.item(),
        #     }
        #     if args.catagorical_progress:
        #         predict_label = torch.argmax(progress_output, dim=1)
        #         accuracy = torch.sum(predict_label == progress) / len(predict_label)
        #         wandb_log["progress_accuracy"] = accuracy

    scaler.scale(loss).backward()
    # loss.backward()
    

    return wandb_log, self_attention_model



def eval_model(args, total_input, mask, batch_triangular_mask, self_attention_model, progress, class_label, 
                classification_loss_function, progress_loss_function):

    progress_output, class_output = self_attention_model(total_input, batch_triangular_mask, mask = mask)

    if args.catagorical_progress:
        progress = progress.long()
    else:
        progress = progress.float().unsqueeze(2)

    if args.two_step_training:
        # batch_size, seq_len, _ = video_array.size()
        class_label = class_label.float()

            # openx_class_label = class_label[:openx_len].view(openx_len * args.max_length)
            # extra_class_label = class_label[openx_len:].view(extra_len * args.max_length)

            # openx_progress_label = progress[:openx_len].view(openx_len * args.max_length)
            # extra_progress_label = progress[openx_len:].view(extra_len * args.max_length)

            # pred_openx_class = class_output[:openx_len].view(openx_len * args.max_length, -1).squeeze(1)
            # pred_extra_class = class_output[openx_len:].view(extra_len * args.max_length, -1).squeeze(1)
            # openx_pred_progress = progress_output[:openx_len].view(openx_len * args.max_length, -1).squeeze(1)
            # extra_pred_progress = progress_output[openx_len:].view(extra_len * args.max_length, -1).squeeze(1)

                
            # openx_class_loss = classification_loss_function(pred_openx_class, openx_class_label)
            # extra_class_loss = classification_loss_function(pred_extra_class, extra_class_label)

            # class_loss = openx_class_loss + extra_class_loss

            # openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
            # extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)

            # progress_loss = openx_progress_loss + extra_progress_loss

            # loss = class_loss + progress_loss

            # openx_none_zero_class = openx_class_label != 0
            # extra_none_zero_class = extra_class_label != 0

            # openx_progress_label_true = openx_progress_label[openx_none_zero_class]
            # extra_progress_label_true = extra_progress_label[extra_none_zero_class]
            # openx_pred_progress_true = openx_pred_progress[openx_none_zero_class]
            # extra_pred_progress_true = extra_pred_progress[extra_none_zero_class]

            # openx_zero_class = openx_class_label == 0
            # extra_zero_class = extra_class_label == 0
            # openx_progress_label_false = openx_progress_label[openx_zero_class]
            # extra_progress_label_false = extra_progress_label[extra_zero_class]
            # openx_pred_progress_false = openx_pred_progress[openx_zero_class]
            # extra_pred_progress_false = extra_pred_progress[extra_zero_class]

            # if args.catagorical_progress:
            #     openx_predict_label_true = torch.argmax(openx_pred_progress_true, dim=1)
            #     extra_predict_label_true = torch.argmax(extra_pred_progress_true, dim=1)
            #     openx_predict_label_false = torch.argmax(openx_pred_progress_false, dim=1)
            #     extra_predict_label_false = torch.argmax(extra_pred_progress_false, dim=1)

            #     openx_progress_accuracy_true = torch.sum(openx_predict_label_true == openx_progress_label_true).item() / len(openx_predict_label_true)
            #     extra_progress_accuracy_true = torch.sum(extra_predict_label_true == extra_progress_label_true).item() / len(extra_predict_label_true)
            #     openx_progress_accuracy_false = torch.sum(openx_predict_label_false == openx_progress_label_false).item() / len(openx_predict_label_false)
            #     extra_progress_accuracy_false = torch.sum(extra_predict_label_false == extra_progress_label_false).item() / len(extra_predict_label_false)
            #     openx_progress_accuracy = torch.sum(openx_predict_label_true == openx_progress_label_true).item() / len(openx_predict_label_true)
            #     extra_progress_accuracy = torch.sum(extra_predict_label_true == extra_progress_label_true).item() / len(extra_predict_label_true)

            
            # pred_openx_class_label = pred_openx_class.round()
            # pred_extra_class_label = pred_extra_class.round()

            # openx_class_accuracy = torch.sum(pred_openx_class_label == openx_class_label).item() / len(pred_openx_class_label)
            # extra_class_accuracy = torch.sum(pred_extra_class_label == extra_class_label).item() / len(pred_extra_class_label)

            # wandb_log = {
            #     "loss/total_loss": loss.item(),
            #     "loss/class_loss": class_loss.item(),
            #     "loss/progress_loss": progress_loss.item(),
            #     "loss/openx_class_loss": openx_class_loss.item(),
            #     "loss/extra_class_loss": extra_class_loss.item(),
            #     "openx/openx_class_loss": openx_class_loss.item(),
            #     "extra/extra_class_loss": extra_class_loss.item(),

            #     "accuracy/openx_two_step_accuracy": openx_class_accuracy,
            #     "accuracy/extra_two_step_accuracy": extra_class_accuracy, 
            #     "openx/openx_two_step_accuracy": openx_class_accuracy,
            #     "extra/extra_two_step_accuracy": extra_class_accuracy,
               
            #     "loss/openx_progress_loss": openx_progress_loss.item(),
            #     "loss/extra_progress_loss": extra_progress_loss.item(),
            #     "openx/openx_progress_loss": openx_progress_loss.item(),
            #     "extra/extra_progress_loss": extra_progress_loss.item(),

            #     "train_accuracy/openx_true_data_accuracy": openx_progress_accuracy_true,
            #     "train_accuracy/extra_true_data_accuracy": extra_progress_accuracy_true,
            #     "train_accuracy/openx_wrong_data_accuracy": openx_progress_accuracy_false,
            #     "train_accuracy/extra_wrong_data_accuracy": extra_progress_accuracy_false,

            #     "openx/openx_true_data_accuracy": openx_progress_accuracy_true,
            #     "extra/extra_true_data_accuracy": extra_progress_accuracy_true,
            #     "openx/openx_wrong_data_accuracy": openx_progress_accuracy_false,
            #     "extra/extra_wrong_data_accuracy": extra_progress_accuracy_false,

            #     "train_accuracy/openx_accuracy": openx_progress_accuracy,
            #     "train_accuracy/extra_accuracy": extra_progress_accuracy,
            #     "openx/openx_accuracy": openx_progress_accuracy,
            #     "extra/extra_accuracy": extra_progress_accuracy,
            # }




