import torch
import wandb
from torch.nn.functional import mse_loss
from utils.train_utils import focal_loss, compute_metrics



def make_train_step_progress_fn(self_attention_model, optimizer, scheduler, args, device):

    def train_step_fn(engine, batch):
        #set to cuda
        openx_data, extra_data = batch
        openx_len = len(openx_data["video_array"])
        extra_len = len(extra_data["video_array"])

        self_attention_model.train()
        optimizer.zero_grad()
        positive_video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
        positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"].squeeze()], dim = 0).to(device).float()              
        positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)
        # positive_progress_mask = torch.ones_like(positive_progress).bool()


        negative_video_array_1 = torch.roll(positive_video_array, extra_len, 0)
        negative_text_array_1 = positive_text_array.clone()
        negative_progress_1 = torch.zeros_like(positive_progress)
        # negative_progress_mask_1 = torch.ones_like(negative_progress_1).bool()

        openx_pos_video_array = torch.cat([positive_video_array[:openx_len], negative_video_array_1[:openx_len]], dim = 0)
        openx_pos_text_array = torch.cat([positive_text_array[:openx_len], negative_text_array_1[:openx_len]], dim = 0)
        openx_pos_progress = torch.cat([positive_progress[:openx_len], negative_progress_1[:openx_len]], dim = 0)
        # openx_pos_progress_mask = torch.cat([positive_progress_mask[:openx_len], negative_progress_mask_1[:openx_len]], dim = 0)
            

        extra_pos_video_array = torch.cat([positive_video_array[openx_len:], negative_video_array_1[openx_len:]], dim = 0)
        extra_pos_text_array = torch.cat([positive_text_array[openx_len:], negative_text_array_1[openx_len:]], dim = 0)
        extra_pos_progress = torch.cat([positive_progress[openx_len:], negative_progress_1[openx_len:]], dim = 0)
        # extra_pos_progress_mask = torch.cat([positive_progress_mask[openx_len:], negative_progress_mask_1[openx_len:]], dim = 0)

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
        progress_pred = self_attention_model(video_embedding, text_array)

        openx_progress_pred = progress_pred[:openx_len]
        extra_progress_pred = progress_pred[openx_len:]

        openx_progress_target = progress[:openx_len]
        extra_progress_target = progress[openx_len:]

        valid_openx_progress_pred = openx_progress_pred[openx_target.bool()]
        valid_openx_progress_target = openx_progress_target[openx_target.bool()]

        valid_extra_progress_pred = extra_progress_pred[extra_target.bool()]
        valid_extra_progress_target = extra_progress_target[extra_target.bool()]

        rest_openx_progress_pred = openx_progress_pred[~openx_target.bool()]
        rest_openx_progress_target = openx_progress_target[~openx_target.bool()]

        rest_extra_progress_pred = extra_progress_pred[~extra_target.bool()]
        rest_extra_progress_target = extra_progress_target[~extra_target.bool()]

        openx_progress_loss = mse_loss(valid_openx_progress_pred[:,1:].squeeze(), valid_openx_progress_target[:,1:])
        extra_progress_loss = mse_loss(valid_extra_progress_pred[:,1:].squeeze(), valid_extra_progress_target[:,1:])
        rest_openx_progress_loss = mse_loss(rest_openx_progress_pred[:,1:].squeeze(), rest_openx_progress_target[:,1:])
        rest_extra_progress_loss = mse_loss(rest_extra_progress_pred[:,1:].squeeze(), rest_extra_progress_target[:,1:])

        total_len = len(openx_progress_pred) + len(extra_progress_pred) + len(rest_openx_progress_pred) + len(rest_extra_progress_pred)

        loss = openx_progress_loss * len(openx_progress_pred) / total_len \
                + extra_progress_loss * len(extra_progress_pred) / total_len \
                + rest_openx_progress_loss * len(rest_openx_progress_pred) / total_len \
                + rest_extra_progress_loss * len(rest_extra_progress_pred) / total_len




        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(self_attention_model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Log all metrics


        wandb_log = {
            "train/progress_loss": loss.item(),
            "train/match_openx_progress_loss": openx_progress_loss.item(),
            "train/match_extra_progress_loss": extra_progress_loss.item(),
            "train/missmatch_openx_progress_loss": rest_openx_progress_loss.item(),
            "train/missmatch_extra_progress_loss": rest_extra_progress_loss.item(),
            "train/total_loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(wandb_log)
        return loss.item()
 
    return train_step_fn
