import os
import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.eval_utils import compute_spearman_correlation_multi_annotations, rank_comparison
from utils.eval_utils import generate_rewind_data, generate_rewind_gif, compute_pearson_correlation_from_sequences
from utils.eval_utils import plot_confusion_matrix_from_predictions, compute_mse_from_sequences, compute_spearman_correlation_from_sequences


os.environ["TOKENIZERS_PARALLELISM"] = "False"




def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce_loss = F.binary_cross_entropy(pred.squeeze(), target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1-pt)**gamma * bce_loss
    return focal_loss.mean()




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

def compute_metrics_multi(args, self_attention_model, threshold, compute_gif = False, epoch = None):

    # for file in os.listdir("./"):
    #     if file.endswith(".pkl"):
    #         os.remove(file)
    confusion_matrix, all_seqs, tasks, text_list = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end.pkl",
        args = args,
        threshold = threshold
    )
    # os.remove("final_rewind_cache_oxe_pos_end.pkl")

    confusion_matrix_1, all_seqs1, _, _ = generate_rewind_data(
            h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
            json_path="new_task_v2.json",
            set_type="eval",
            rewind_model=self_attention_model,
            cache_path="final_rewind_cache_oxe_pos_end_1.pkl",
            args = args,
            annotation = 1,
            threshold = threshold
        )
    # os.remove("final_rewind_cache_oxe_pos_end_1.pkl")

    confusion_matrix_2, all_seqs2, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_2.pkl",
        args = args,
        annotation = 2,
        threshold = threshold
    )
    # os.remove("final_rewind_cache_oxe_pos_end_2.pkl")

    confusion_matrix_3, all_seqs3, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_3.pkl",
        args = args,
        annotation = 3,
        threshold = threshold
    )
    # os.remove("final_rewind_cache_oxe_pos_end_3.pkl")

    confusion_matrix_all_fail, _, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval_all_fail.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_fail.pkl",
        args = args,
        threshold = threshold
    )
    # os.remove("final_rewind_cache_oxe_pos_end_fail.pkl")

    confusion_matrix_close_success, _, _, _ = generate_rewind_data(
        h5_path="eval_rewind/metaworld_dino_embeddings_eval_close_succ.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=self_attention_model,
        cache_path="final_rewind_cache_oxe_pos_end_close_succ.pkl",
        args = args,
        threshold = threshold
    )
    # os.remove("final_rewind_cache_oxe_pos_end_close_succ.pkl")


    compute_pearson_correlation_from_sequences(
        all_seqs=all_seqs,
        set_type="eval",
        project_name="roboclip-v2",
        env_names=tasks,
        threshold=threshold,
        epoch=epoch
    )

    plot_confusion_matrix_from_predictions(
        predicted_rewards=confusion_matrix,
        task_names=tasks,
        set_type="eval",
        text_instructions=text_list,
        fig_name="Rewind",
        threshold=threshold,
        epoch=epoch
    )


    # # ============ 4) 计算 MSE ============
    compute_mse_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval",
        threshold=threshold,
        epoch=epoch
    )

    # ============ 5) 计算 Spearman 相关系数 ============
    compute_spearman_correlation_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval",
        threshold=threshold,
        epoch=epoch
    )

    compute_spearman_correlation_multi_annotations(
        all_seqs_a=all_seqs1,
        all_seqs_b=all_seqs2,
        all_seqs_c=all_seqs3,
        all_seqs_d=all_seqs,
        env_names=tasks,
        set_type="eval",
        threshold=threshold,
        epoch=epoch
    )

    rank_comparison(confusion_matrix_all_fail, confusion_matrix_close_success, confusion_matrix, tasks, threshold, epoch=epoch)


    if compute_gif:
        print("Generating GIFs,generate_rewind_gif", epoch)
        generate_rewind_gif(
            h5_path="eval_rewind/metaworld_dino_embeddings_eval_close_succ_128.h5",
            json_path="new_task_v2.json",
            set_type="eval",
            rewind_model=self_attention_model,
            device="cuda",
            args=args,
            threshold=threshold,
            epoch=epoch,
            suboptimal_type="close_success",
        )

        generate_rewind_gif(
            h5_path="eval_rewind/metaworld_dino_embeddings_eval_all_fail_128.h5",
            json_path="new_task_v2.json",
            set_type="eval",
            rewind_model=self_attention_model,
            device="cuda",
            args=args,
            threshold=threshold,
            epoch=epoch,
            suboptimal_type="all_fail",
        )


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

