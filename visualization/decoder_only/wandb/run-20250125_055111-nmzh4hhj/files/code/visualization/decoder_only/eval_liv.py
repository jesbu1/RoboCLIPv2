import h5py
import torch
from clip_utils import normalize_embeddings
import json
import random
import numpy as np
from clip_utils import load_model, embedding_text, embedding_image, SingleLayerMLP
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import io
from PIL import Image
import os
from typing import List, Optional
import pickle
import textwrap
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def shorten_name(name, separator=" ", max_length=5):
    parts = name.split(separator)
    return separator.join([part[:max_length] for part in parts])
    
def shorten_text(text, max_length=25):
    print(f"og_text: {text}")
    shortened_text = textwrap.shorten(text, width=max_length, placeholder="...")
    print(f"shortened_text: {shortened_text}")
    return shortened_text

def plot_matrix_as_image(matrix, names, set, text):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.25, len(matrix) * 1))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)

    # 只保留两位小数
    cbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
    cbar.update_ticks()  # 更新刻度标签

    # 放大颜色条字体
    cbar.ax.yaxis.set_tick_params(labelsize=16)  # 你可以调整 `fontsize`

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    # shortened_text = [shorten_name(name, max_length = 6) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 6) for name in names]

    shortened_text = [shorten_text(name, max_length = 25) for name in text]
    shortened_names = [shorten_name(name, separator = "-", max_length = 12) for name in names]

    # Label each row and column with the given names
    ax.set_xticklabels(shortened_text, rotation=30, ha='left', fontsize=18)
    ax.set_yticklabels(shortened_names, fontsize=18)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=20)
    # keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    wandb.log({f"confusion_matrix/{set}_confusion_matrix": wandb.Image(fig)})
    plt.savefig(f"confusion_matrix_{set}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory


def generate_liv_data(
    h5_path: str,
    json_path: str,
    set_type: str,
    device: str = "cuda",
    cache_path="liv_cache.pkl"
):
    """
    与 generate_gemini_data 类似，遍历 (环境, 文本) 组合，生成:
      1) (N x N) 混淆矩阵 (最后帧单值)
      2) (N 条完整序列)，只对 i==j 的组合做“逐帧”或“增量”推理

    返回:
        (confusion_matrix, predicted_sequences, tasks, text_list)
          confusion_matrix: shape=(N,N), float，表示 [0..1] reward
          predicted_sequences: List[List[float]], 第 i 项为该环境 i 的完整序列
          tasks: 任务名称列表
          text_list: 文本说明列表
    """

    # 1) 从 new_task_v2.json 中读取 tasks
    task_subset = json.load(open(json_path, "r"))
    if set_type == "train":
        tasks = task_subset["training_tasks"]
    elif set_type == "eval":
        tasks = task_subset["eval_tasks"]
    else:
        tasks = task_subset["test_tasks"]
    num_tasks = len(tasks)

    liv, processor, tokenizer = load_model("liv")
    liv = liv.to(device)

    # 2) 打开 HDF5
    with h5py.File(h5_path, "r") as f:
        
        # 为混淆矩阵 (N x N) 分配空间
        confusion_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        # 仅对 i==j 的“匹配环境+文本”做整段序列 => predicted_sequences[i]
        predicted_sequences: List[Optional[List[float]]] = [None]*num_tasks

        # 3) 读取所有文本（与 tasks 对应）。构造 text_list
        #    row i => video tasks[i], col j => text_list[j]
        text_list = []
        for env_name in tasks:
            text_dset_name = f"text_annotations/{env_name}_text"
            if text_dset_name not in f:
                raise KeyError(f"HDF5 中不存在文本数据集 {text_dset_name}")
            text_bytes = f[text_dset_name][()]
            text_str = text_bytes.decode("utf-8")
            print(f"📝 {env_name}: {text_str}")
            text_list.append(text_str)

        # ====== 读/写 缓存 ======
        cache_dict = {}
        if os.path.exists(cache_path):
            print(f"[Cache] Loading cache from {cache_path} ...")
            with open(cache_path, "rb") as cf:
                cache_dict = pickle.load(cf)
            print(f"[Cache] Already have {len(cache_dict)} items in cache.")

        def save_cache():
            """将 cache_dict 写回 pkl 文件"""
            with open(cache_path, "wb") as cf:
                pickle.dump(cache_dict, cf)

        # 4) 初始化每个(环境)的 VLC 类(可在循环里初始化，或先不初始化)
        #    这里每次都重新 new SingleVideoVLCRewardCalculator，
        #    并传入 caption_text。也可只做 1 个对象，在 compute_video_reward 前改 caption_text。


        # 5) 遍历所有 (i, j)
        for i, env_video_name in enumerate(tasks):
            # 5.1) 读取该环境对应的视频帧 => shape (frame_count, 224,224,3) or (frame_count, H, W, 3)
            #      这里假设 HDF5 中存的就是 (N, H, W, 3) (RGB)
            frames_data = f[env_video_name][:]  # np.ndarray
            total_frames = frames_data.shape[0]

            # 5.2) 遍历所有文本 j
            for j, text_str in enumerate(text_list):
                cache_key_full = (env_video_name, text_str, total_frames)

                if cache_key_full in cache_dict:
                    # 已缓存 => 直接用
                    video_reward = cache_dict[cache_key_full]
                else:
                    # ============ A) 计算混淆矩阵中 (i,j) 的最后帧单值 ============
                    # 先将 frames_data 转成 video_tensor 并 compute_video_reward
                    # singleVideoCalc.transform_frames => torch.Tensor => compute_video_reward
                    # 注意: compute_video_reward() 内部会自动做抽帧到 self.max_frames
                    #       并返回 [0..1] reward
                    frames_embed = embedding_image(liv, processor, Image.fromarray(frames_data[-1].astype(np.uint8)))

                    text_embed = embedding_text(liv, tokenizer, text_str)
                    print(frames_embed.shape, text_embed.shape)
                    video_reward = F.cosine_similarity(frames_embed, text_embed, dim=1).detach().cpu().item()

                    cache_dict[cache_key_full] = video_reward
                    save_cache()
                print(f"🎬 {env_video_name} ({text_str}): {video_reward}")
                confusion_matrix[i, j] = video_reward

                # ============ B) 如果 i==j => 逐帧/增量生成完整序列 ============
                if i == j:
                    # 这里示例做“增量序列”：从帧 1..N，每次 compute_video_reward
                    # 也可以只在 full frames 上调一次 => predicted_sequences[i] = [video_reward]
                    # 但是为了和 gemini_data “逐帧相关性”一致，示例如下:
                    predicted_seq = []
                    if total_frames < 1:
                        predicted_seq = []
                    else:
                        all_frames_embed = torch.stack([
                                    embedding_image(liv, processor, Image.fromarray(frame.astype(np.uint8))).squeeze()
                                    for frame in frames_data
                                ], dim=0)
                        # print(sub_frames_embed.shape)
                        text_embed = embedding_text(liv, tokenizer, text_str)
                        # 对 partial_count in [1..total_frames]
                        for partial_count in range(1, total_frames + 1):
                            cache_key_partial = (env_video_name, text_str, partial_count)
                            if cache_key_partial in cache_dict:
                                sub_reward = cache_dict[cache_key_partial]
                            else:
                                # 需要计算
                                # sub_frames => frames_data[:partial_count]
                                sub_frames_embed = all_frames_embed[partial_count - 1].unsqueeze(0)
                                # print(sub_frames_embed.shape, text_embed.shape)
                                sub_reward = F.cosine_similarity(sub_frames_embed, text_embed, dim=1).detach().cpu().item()
                                cache_dict[cache_key_partial] = sub_reward
                                save_cache()

                            predicted_seq.append(sub_reward)
                            
                    print(f"🎬 {env_video_name} ({text_str}): {predicted_seq}")
                    

                    predicted_sequences[i] = predicted_seq
                    print(f"🎬 : {predicted_sequences[i]}")

        return confusion_matrix, predicted_sequences, tasks, text_list


def compute_pearson_correlation_from_sequences(
    all_seqs,
    env_names,
    set_type: str,
    project_name: str = "roboclip-v2"
):
    """
    给定所有环境的完整预测序列 (all_seqs)，以及环境名称 (env_names)，
    分别计算与参考序列 (1..len(seq)) 的 Pearson 相关系数，
    并将每个环境的相关系数及平均值上传到 wandb。

    :param all_seqs:   List of length N, 
                       all_seqs[i] 是第 i 个环境的逐帧预测值 (0..100), 形如 [val0, val1, ...]
    :param env_names:  List of length N, 
                       env_names[i] 为第 i 个环境（或任务）的名称 (str)
    :param set_type:   "train" 或 "eval" 等标识，用于 wandb log
    :param project_name: wandb 项目名称
    :return: (avg_corr, correlations)
             avg_corr: 所有环境的平均 Pearson
             correlations: 长度 N 的列表，每个环境对应一个 Pearson 值
    """
    if len(all_seqs) != len(env_names):
        print("[!] all_seqs 和 env_names 长度不一致，无法一一对应。")
        return 0.0, []

    

    correlations = []

    for i, seq in enumerate(all_seqs):
        env_name = env_names[i]
        if len(seq) < 2:
            # 如果帧数<2，无法计算Pearson，设为0
            corr_val = 0.0
        else:
            # 将 [0..100] 转为 [0..1]
            pred_array = np.array(seq, dtype=np.float32)
            n = len(pred_array)
            # 构造参考序列 [1..n]
            gt_array = np.linspace(1, n, n, dtype=np.float32)

            corr_val, _ = pearsonr(pred_array, gt_array)

        correlations.append(corr_val)
        # 逐个 log 到 wandb
        wandb.log({f"{set_type}_pearson_correlation/{env_name}_pearson": corr_val})

    # 计算平均
    if len(correlations) > 0:
        avg_corr = float(np.mean(correlations))
    else:
        avg_corr = 0.0

    wandb.log({f"{set_type}_pearson_correlation/average_pearson": avg_corr})

    print(f"[{set_type}] 平均Pearson相关系数: {avg_corr:.4f}")
    return avg_corr, correlations


def plot_confusion_matrix_from_predictions(
    predicted_rewards: np.ndarray,
    task_names: list,
    text_instructions: list,
    set_type: str
):
    """
    将 NxN 矩阵 predicted_rewards 以混淆矩阵形式绘制。

    :param predicted_rewards: (N x N) 矩阵
    :param task_names:       行标签，对应任务名
    :param text_instructions:列标签，对应文本指令
    :param set_type:         'train', 'eval' 或其他标识字符串
    """
    # 只要把 predicted_rewards 作为 matrix,
    # task_names 作为 names (行标签),
    # text_instructions 作为 text (列标签),
    # set_type 作为 set,
    # 传给你已定义的函数即可：

    plot_matrix_as_image(
        matrix=predicted_rewards, 
        names=task_names,
        set=set_type,
        text=text_instructions
    )


def compute_mse_from_sequences(
    all_seqs,
    env_names,
    set_type: str,
):
    """
    计算 all_seqs 中每个环境的预测值与参考序列 (1..len(seq)) 之间的 MSE，
    并分别计算 Overall MSE (所有帧) 和 Final-frame MSE (仅最后一帧)，并绘制曲线到 wandb。

    :param all_seqs:   List of length N, 
                       all_seqs[i] 是第 i 个环境的逐帧预测值 (0..100), 形如 [val0, val1, ...]
    :param env_names:  List of length N, 
                       env_names[i] 为第 i 个环境（或任务）的名称 (str)
    :param set_type:   "train" 或 "eval" 等标识，用于 wandb log
    :return: (avg_mse, avg_final_mse, mse_list, final_mse_list)
             avg_mse: 所有环境的 Overall MSE 均值
             avg_final_mse: 所有环境的 Final-frame MSE 均值
             mse_list: 长度 N 的列表，每个环境对应一个 Overall MSE
             final_mse_list: 长度 N 的列表，每个环境对应一个 Final-frame MSE
    """
    if len(all_seqs) != len(env_names):
        print("[!] all_seqs 和 env_names 长度不一致，无法一一对应。")
        return 0.0, 0.0, [], []

    mse_list = []
    final_mse_list = []

    for i, seq in enumerate(all_seqs):
        env_name = env_names[i]
        if len(seq) < 2:
            # 如果帧数 < 2，无法计算 MSE，设为 0
            mse_val = 0.0
            final_mse_val = 0.0
        else:
            pred_array = np.array(seq, dtype=np.float32)
            n = len(pred_array)

            gt_array = np.linspace(0, 1, n, dtype=np.float32)
            frame_numbers = np.arange(n)  # Frame Index

            # 计算 Overall MSE
            mse_val = mean_squared_error(gt_array, pred_array)

            # 计算 Final-frame MSE (只看最后一帧)
            final_mse_val = mean_squared_error([gt_array[-1]], [pred_array[-1]])

        mse_list.append(mse_val)
        final_mse_list.append(final_mse_val)

        # 逐个 log 到 wandb
        wandb.log({f"{set_type}_overall_mse/{env_name}": mse_val})
        wandb.log({f"{set_type}_final_mse/{env_name}": final_mse_val})

        # 🔹 绘制预测 vs 真实值曲线
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frame_numbers, gt_array, label="Ground Truth", linestyle="dashed", color="blue")
        ax.plot(frame_numbers, pred_array, label="Prediction", linestyle="-", color="red")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Reward")
        ax.set_title(f"{env_name} Prediction vs. GT")
        ax.legend()
        plt.tight_layout()

        # 🔹 记录到 wandb
        wandb.log({f"{set_type}_curve/{env_name}": wandb.Image(fig)})

        plt.close(fig)  # 释放内存

    # 计算所有环境的均值
    avg_mse = float(np.mean(mse_list)) if mse_list else 0.0
    avg_final_mse = float(np.mean(final_mse_list)) if final_mse_list else 0.0

    # Log 平均值
    wandb.log({f"{set_type}_mse/average_overall_mse": avg_mse})
    wandb.log({f"{set_type}_mse/average_final_mse": avg_final_mse})

    print(f"[{set_type}] 平均 Overall MSE: {avg_mse:.4f}")
    print(f"[{set_type}] 平均 Final-frame MSE: {avg_final_mse:.4f}")

    return avg_mse, avg_final_mse, mse_list, final_mse_list


def compute_spearman_correlation_from_sequences(
    all_seqs,
    env_names,
    set_type: str,
):
    """
    给定所有环境的完整预测序列 (all_seqs)，以及环境名称 (env_names)，
    分别计算与参考序列 (1..len(seq)) 的 Pearson 相关系数，
    并将每个环境的相关系数及平均值上传到 wandb。

    :param all_seqs:   List of length N, 
                       all_seqs[i] 是第 i 个环境的逐帧预测值 (0..100), 形如 [val0, val1, ...]
    :param env_names:  List of length N, 
                       env_names[i] 为第 i 个环境（或任务）的名称 (str)
    :param set_type:   "train" 或 "eval" 等标识，用于 wandb log
    :param project_name: wandb 项目名称
    :return: (avg_corr, correlations)
             avg_corr: 所有环境的平均 Pearson
             correlations: 长度 N 的列表，每个环境对应一个 Pearson 值
    """
    if len(all_seqs) != len(env_names):
        print("[!] all_seqs 和 env_names 长度不一致，无法一一对应。")
        return 0.0, []

    

    correlations = []

    for i, seq in enumerate(all_seqs):
        env_name = env_names[i]
        if len(seq) < 2:
            # 如果帧数<2，无法计算Pearson，设为0
            corr_val = 0.0
        else:
            # 将 [0..100] 转为 [0..1]
            pred_array = np.array(seq, dtype=np.float32)
            n = len(pred_array)
            # 构造参考序列 [1..n]
            gt_array = np.linspace(1, n, n, dtype=np.float32)

            corr_val, _ = spearmanr(pred_array, gt_array)

        correlations.append(corr_val)
        # 逐个 log 到 wandb
        wandb.log({f"{set_type}_spearman_correlation/{env_name}_spearman": corr_val})

    # 计算平均
    if len(correlations) > 0:
        avg_corr = float(np.mean(correlations))
    else:
        avg_corr = 0.0

    wandb.log({f"{set_type}_spearman_corrleation/average_spearman": avg_corr})

    print(f"[{set_type}] 平均Spearman相关系数: {avg_corr:.4f}")
    return avg_corr, correlations



if __name__ == "__main__":
    confusion_matrix, predicted_sequences, tasks, text_list = generate_liv_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_GT_eval_v2.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        device="cuda",
        cache_path="liv_cache.pkl"
    )
    wandb.init(project="roboclip-v2", name=f"eval_GVL")

    # ============ 2) 计算相关(只用对角线) ============
    compute_pearson_correlation_from_sequences(
        all_seqs=predicted_sequences,
        set_type="eval",
        project_name="roboclip-v2",
        env_names=tasks
    )

    # ============ 3) 绘制混淆矩阵 =============
    plot_confusion_matrix_from_predictions(
        predicted_rewards=confusion_matrix,
        task_names=tasks,
        set_type="eval",
        text_instructions=text_list
    )


    # ============ 4) 计算 MSE ============
    compute_mse_from_sequences(
        all_seqs=predicted_sequences,
        env_names=tasks,
        set_type="eval"
    )

    # ============ 5) 计算 Spearman 相关系数 ============
    compute_spearman_correlation_from_sequences(
        all_seqs=predicted_sequences,
        env_names=tasks,
        set_type="eval"
    )

    wandb.finish()

    
    


