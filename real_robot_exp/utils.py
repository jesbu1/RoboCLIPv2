import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
import math
from tqdm import tqdm
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




def update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label, 
                classification_loss_function, progress_loss_function, optimizer, openx_len = None, extra_len = None, scheduler = None):
    
    progress_output, class_output = self_attention_model(video_array, batch_triangular_mask, text_array, mask = None)
    
    if args.catagorical_progress:
        progress = progress.long()
    else:
        progress = progress.float().unsqueeze(2)

    if args.two_step_training:
        batch_size, seq_len, _ = video_array.size()
        class_label = class_label.long()

        if openx_len is not None:
            openx_class_label = class_label[:openx_len]
            extra_class_label = class_label[openx_len:]
            openx_progress_label = progress[:openx_len]
            extra_progress_label = progress[openx_len:]

            pred_openx_class = class_output[:openx_len].view(openx_len * seq_len, -1).squeeze(1)
            pred_extra_class = class_output[openx_len:].view(extra_len * seq_len, -1).squeeze(1)
            openx_pred_progress = progress_output[:openx_len]
            extra_pred_progress = progress_output[openx_len:]

            openx_class_label = openx_class_label.view(openx_len * seq_len)
            extra_class_label = extra_class_label.view(extra_len * seq_len)
            
            openx_class_loss = classification_loss_function(pred_openx_class, openx_class_label)
            extra_class_loss = classification_loss_function(pred_extra_class, extra_class_label)

            class_loss = openx_class_loss + extra_class_loss

            openx_none_zero_class = openx_class_label != 0
            extra_none_zero_class = extra_class_label != 0
            if args.progress_loss:
                if not args.catagorical_progress:
                    openx_progress_label = openx_progress_label.view(openx_len * seq_len)[openx_none_zero_class]
                    extra_progress_label = extra_progress_label.view(extra_len * seq_len)[extra_none_zero_class]
                    openx_pred_progress = progress_output[:openx_len].view(openx_len * seq_len, -1).squeeze(1)[openx_none_zero_class]
                    extra_pred_progress = progress_output[openx_len:].view(extra_len * seq_len, -1).squeeze(1)[extra_none_zero_class]

                    openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
                    extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)
                else:

                    openx_progress_label = openx_progress_label.view(openx_len * seq_len)[openx_none_zero_class]
                    extra_progress_label = extra_progress_label.view(extra_len * seq_len)[extra_none_zero_class]
                    openx_pred_progress = progress_output[:openx_len].view(openx_len * seq_len, -1).squeeze(1)[openx_none_zero_class]
                    extra_pred_progress = progress_output[openx_len:].view(extra_len * seq_len, -1).squeeze(1)[extra_none_zero_class]

                    openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
                    extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)

                progress_loss = openx_progress_loss + extra_progress_loss

                loss = class_loss + progress_loss
            else:
                loss = class_loss


            openx_class_predict_label = torch.argmax(pred_openx_class, dim=1)
            extra_class_predict_label = torch.argmax(pred_extra_class, dim=1)

            openx_class_accuracy = torch.sum(openx_class_predict_label == openx_class_label).item() / len(openx_class_predict_label)
            extra_class_accuracy = torch.sum(extra_class_predict_label == extra_class_label).item() / len(extra_class_predict_label)

            openx_class_none_zero = openx_class_label != 0
            openx_class_zero = openx_class_label == 0

            extra_class_none_zero = extra_class_label != 0
            extra_class_zero = extra_class_label == 0

            openx_class_none_zero_accuracy = torch.sum(openx_class_predict_label[openx_class_none_zero] == openx_class_label[openx_class_none_zero]).item() / len(openx_class_predict_label[openx_class_none_zero])
            openx_class_zero_accuracy = torch.sum(openx_class_predict_label[openx_class_zero] == openx_class_label[openx_class_zero]).item() / len(openx_class_predict_label[openx_class_zero])

            extra_class_none_zero_accuracy = torch.sum(extra_class_predict_label[extra_class_none_zero] == extra_class_label[extra_class_none_zero]).item() / len(extra_class_predict_label[extra_class_none_zero])
            extra_class_zero_accuracy = torch.sum(extra_class_predict_label[extra_class_zero] == extra_class_label[extra_class_zero]).item() / len(extra_class_predict_label[extra_class_zero])



            wandb_log = {
                "total_loss": loss.item(),
                "class_loss": class_loss.item(),
                "openx_class_loss": openx_class_loss.item(),
                "extra_class_loss": extra_class_loss.item(),
                "openx_class_accuracy": openx_class_accuracy,
                "extra_class_accuracy": extra_class_accuracy,
                "train_accuracy/openx_true_data_accuracy": openx_class_none_zero_accuracy,
                "train_accuracy/openx_wrong_data_accuracy": openx_class_zero_accuracy,
                "train_accuracy/extra_true_data_accuracy": extra_class_none_zero_accuracy,
                "train_accuracy/extra_wrong_data_accuracy": extra_class_zero_accuracy
                
            }

            if args.progress_loss:
                wandb_log["progress_loss"] = progress_loss.item()
                wandb_log["openx_progress_loss"] = openx_progress_loss.item()
                wandb_log["extra_progress_loss"] = extra_progress_loss.item()

            if args.catagorical_progress:

                openx_predict_label = torch.argmax(openx_pred_progress, dim=1)
                extra_predict_label = torch.argmax(extra_pred_progress, dim=1)
                openx_progress_accuracy = torch.sum(openx_predict_label == openx_progress_label).item() / len(openx_predict_label)
                extra_progress_accuracy = torch.sum(extra_predict_label == extra_progress_label).item() / len(extra_predict_label)
                wandb_log["openx_progress_accuracy"] = openx_progress_accuracy
                wandb_log["extra_progress_accuracy"] = extra_progress_accuracy

        else:
        
            class_label = class_label.view(batch_size * seq_len)
            progress = progress.view(batch_size * seq_len, -1)

            class_output = class_output.view(batch_size * seq_len, -1)
            progress_output = progress_output.view(batch_size * seq_len, -1)
            class_loss = classification_loss_function(class_output, class_label)

            none_zero_class = class_label != 0
            if args.progress_loss:
                if not args.catagorical_progress:
                    progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class])
                else:
                    progress = progress[none_zero_class].squeeze(1)
                    progress_output = progress_output[none_zero_class]
                    progress_loss = progress_loss_function(progress_output, progress - 1)
                loss = class_loss + progress_loss
            else:
                loss = class_loss
            class_predict_label = torch.argmax(class_output, dim=1)
            class_accuracy = torch.sum(class_predict_label == class_label).item() / len(class_predict_label)

            wandb_log = {
                "total_loss": loss.item(),
                "class_loss": class_loss.item(),
                "class_accuracy": class_accuracy
            }
            if args.progress_loss:
                wandb_log["progress_loss"] = progress_loss.item()
            
            if args.catagorical_progress:
                predict_label = torch.argmax(progress_output, dim=1)
                progress_accuracy = torch.sum(predict_label == progress).item() / len(predict_label)
                wandb_log["progress_accuracy"] = progress_accuracy

    else:
        batch_size, seq_len, _ = video_array.size()

        if openx_len is not None:
            openx_progress_label = progress[:openx_len]
            extra_progress_label = progress[openx_len:]

            openx_pred_progress = progress_output[:openx_len]
            extra_pred_progress = progress_output[openx_len:]

            if args.progress_loss:
                if not args.catagorical_progress:
                    openx_progress_label = openx_progress_label.view(openx_len * seq_len)
                    extra_progress_label = extra_progress_label.view(extra_len * seq_len)
                    openx_pred_progress = progress_output[:openx_len].view(openx_len * seq_len, -1).squeeze(1)
                    extra_pred_progress = progress_output[openx_len:].view(extra_len * seq_len, -1).squeeze(1)

                    openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
                    extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)
                else:

                    openx_progress_label = openx_progress_label.view(openx_len * seq_len)
                    extra_progress_label = extra_progress_label.view(extra_len * seq_len)
                    openx_pred_progress = progress_output[:openx_len].view(openx_len * seq_len, -1).squeeze(1)
                    extra_pred_progress = progress_output[openx_len:].view(extra_len * seq_len, -1).squeeze(1)

                    openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
                    extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)

                progress_loss = openx_progress_loss + extra_progress_loss

                loss = progress_loss


            # wandb_log["total_loss"] = progress_loss.item()
            # wandb_log["openx_progress_loss"] = openx_progress_loss.item()
            # wandb_log["extra_progress_loss"] = extra_progress_loss.item()
            wandb_log = {
                "total_loss": loss.item(),
                "openx_progress_loss": openx_progress_loss.item(),
                "extra_progress_loss": extra_progress_loss.item(),
            }

            if args.catagorical_progress:

                openx_predict_label = torch.argmax(openx_pred_progress, dim=1)
                extra_predict_label = torch.argmax(extra_pred_progress, dim=1)
                openx_progress_accuracy = torch.sum(openx_predict_label == openx_progress_label).item() / len(openx_predict_label)
                extra_progress_accuracy = torch.sum(extra_predict_label == extra_progress_label).item() / len(extra_predict_label)
                wandb_log["openx_progress_accuracy"] = openx_progress_accuracy
                wandb_log["extra_progress_accuracy"] = extra_progress_accuracy




    optimizer.zero_grad()
    loss.backward()
    if args.clip_grad:
        torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

        
    wandb_log["lr"] = optimizer.param_groups[0]["lr"]
    return wandb_log, self_attention_model


def eval_model(positive_eval_openx_dataset, self_attention_model, progress_loss_function, triangular_mask, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct_num = 0
    total_num = 0
    total_loss = 0
    # wrong_num = 0
    for eval_data in tqdm(positive_eval_openx_dataset):
        
        video_array = eval_data["video_array"].to(device).float()
        text_array = eval_data["text_array"].to(device).float().squeeze(1)
        progress = eval_data["progress"].to(device)
        if args.catagorical_progress:
            progress = progress.long()

        eval_batch_size, seq_len, _ = video_array.size()
        eval_triangular_mask = triangular_mask.repeat(eval_batch_size, 1, 1, 1).bool()
        progress_output, class_output = self_attention_model(video_array, eval_triangular_mask, text_array, mask = None)
        
        progress = progress.view(eval_batch_size * seq_len, -1)
        progress_output = progress_output.view(eval_batch_size * seq_len, -1)

        if args.two_step_training:
            class_label = eval_data["class_label"].to(device)               
            class_label = class_label.view(eval_batch_size * seq_len)
            class_output = class_output.view(eval_batch_size * seq_len, -1)
            none_zero_class = class_label != 0
            if not args.catagorical_progress:
                progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class])
            else:
                progress_loss = progress_loss_function(progress_output[none_zero_class], progress[none_zero_class].long().squeeze(1))
            class_predict_label = torch.argmax(class_output, dim=1)
            # class_accuracy = torch.sum(class_predict_label == class_label).item() / len(class_predict_label)
            correct_num += torch.sum(class_predict_label == class_label).item()
            

        else:
            if not args.catagorical_progress:
                progress_loss = progress_loss_function(progress_output, progress)
            else:
                progress_loss = progress_loss_function(progress_output, progress.long().squeeze(1))

            
        total_num += len(progress)
        total_loss += progress_loss.item()
    
    progress = total_loss / total_num

    if args.two_step_training:
        class_accuracy = correct_num / total_num
    else:
        class_accuracy = None

    return class_accuracy, progress

