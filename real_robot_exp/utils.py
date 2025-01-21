import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def update_model(args, video_array, text_array, batch_triangular_mask, self_attention_model, progress, class_label, 
                classification_loss_function, progress_loss_function, optimizer, openx_len = None, extra_len = None):
    # import pdb; pdb.set_trace()
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

                openx_progress_label = openx_progress_label.view(openx_len * seq_len)[openx_none_zero_class]
                extra_progress_label = extra_progress_label.view(extra_len * seq_len)[extra_none_zero_class]
                openx_pred_progress = progress_output[:openx_len].view(openx_len * seq_len, -1).squeeze(1)[openx_none_zero_class]
                extra_pred_progress = progress_output[openx_len:].view(extra_len * seq_len, -1).squeeze(1)[extra_none_zero_class]

                openx_progress_loss = progress_loss_function(openx_pred_progress, openx_progress_label)
                extra_progress_loss = progress_loss_function(extra_pred_progress, extra_progress_label)

                progress_loss = openx_progress_loss + extra_progress_loss

                loss = class_loss + progress_loss

                openx_class_predict_label = torch.argmax(pred_openx_class, dim=1)
                extra_class_predict_label = torch.argmax(pred_extra_class, dim=1)

                openx_class_accuracy = torch.sum(openx_class_predict_label == openx_class_label).item() / len(openx_class_predict_label)
                extra_class_accuracy = torch.sum(extra_class_predict_label == extra_class_label).item() / len(extra_class_predict_label)

                wandb_log = {
                    "total_loss": loss.item(),
                    "class_loss": class_loss.item(),
                    "progress_loss": progress_loss.item(),
                    "openx_class_loss": openx_class_loss.item(),
                    "extra_class_loss": extra_class_loss.item(),
                    "openx_class_accuracy": openx_class_accuracy,
                    "extra_class_accuracy": extra_class_accuracy,
                    "openx_progress_loss": openx_progress_loss.item(),
                    "extra_progress_loss": extra_progress_loss.item()
                }

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
            progress = progress.view(batch_size * seq_len, -1)
            if args.catagorical_progress:
                progress = progress.squeeze(1)
            loss = progress_loss_function(progress_output, progress)
            wandb_log = {
                "progress_loss": loss.item(),
            }
            if args.catagorical_progress:
                predict_label = torch.argmax(progress_output, dim=1)
                accuracy = torch.sum(predict_label == progress) / len(predict_label)
                wandb_log["progress_accuracy"] = accuracy


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return wandb_log, self_attention_model

