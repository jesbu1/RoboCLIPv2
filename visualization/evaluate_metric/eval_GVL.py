import numpy as np
import json
import h5py
from typing import Dict, List, Union, Optional
from scipy.stats import pearsonr, spearmanr
import wandb
import matplotlib.pyplot as plt
from GVL import GeminiVideoAnalyzerHDF5
import pickle
import os
import requests
import time
import textwrap
from sklearn.metrics import mean_squared_error
from rank_comparsion import rank_comparison

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
    m_min = matrix.min()
    m_max = matrix.max()
    if m_max == m_min:
        # 说明整张矩阵所有值相同，可以直接都置为0 或 1
        # 这里演示直接设置为 0
        matrix= np.zeros_like(matrix)
    else:
        matrix = (matrix - m_min) / (m_max - m_min)

    # 只保留两位小数
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.25, len(matrix) * 1))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    # cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)

    # # 只保留两位小数
    # cbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
    # cbar.update_ticks()  # 更新刻度标签

    # # 放大颜色条字体
    # cbar.ax.yaxis.set_tick_params(labelsize=16)  # 你可以调整 `fontsize`

    # Set x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # shortened_text = [shorten_name(name, max_length = 6) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 6) for name in names]

    shortened_text = [shorten_text(name, max_length = 25) for name in text]
    shortened_names = [shorten_name(name, separator = "-", max_length = 12) for name in names]

    # Label each row and column with the given names
    # ax.set_xticklabels(shortened_text, rotation=30, ha='left', fontsize=18)
    # ax.set_yticklabels(shortened_names, fontsize=18)

    # Display the values in the matrix
    # for (i, j), val in np.ndenumerate(matrix):
    #     ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=20)

    # keep 2 digit first 2 digit after decimal point {val:.2f}
    # Adjust layout to fit labels
    plt.tight_layout()

    # Convert Matplotlib figure to PIL Image
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    wandb.log({f"confusion_matrix/{set}_confusion_matrix_GVL": wandb.Image(fig)})
    plt.savefig(f"confusion_matrix_{set}_GVL.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory



def generate_gemini_data(
    h5_path: str,
    json_path: str,
    set_type: str,
    gemini_api_key: str,
    analyzer_class,
    max_frames: int = 15,
    offset: float = 0.5,
    max_retries: int = 10,
    base_delay: float = 3.0,
    cache_path: Optional[str] = None
):
    """
    同时生成:
      1) (N x N) 混淆矩阵: 对 (视频 env_i, 文本 env_j) 取推理结果的最后帧 (归一化至 [0..1])
      2) (N 条完整序列): 对每个环境 i，仅在 (env_i, text_i) 时，保存整段预测序列(做相关性分析, [0..1])

    启用指数退避：当 run_analysis() 遇到 HTTP 429 时，等待 base_delay * 2^attempt 再重试。
    若重试 max_retries 次后仍失败，则抛出异常并中断。

    缓存机制：
      - 若提供 cache_path，则使用它来记录已经完成的 (env, text) => completion_array。
      - 当再次运行时，若已在 cache 中找到该对 (env, text)，则直接跳过请求，使用缓存结果。

    :param h5_path:       HDF5 文件路径
    :param json_path:     new_task_v2.json，含 {"training_tasks": [...], "eval_tasks": [...], "test_tasks": [...]}
    :param set_type:      "train" / "eval" / "test"
    :param gemini_api_key:Gemini API Key
    :param analyzer_class:类似 GeminiVideoAnalyzerHDF5，拥有 run_analysis() => List[float]
    :param max_frames:    每段视频采样帧数
    :param offset:        采样帧时的偏移
    :param max_retries:   重试次数
    :param base_delay:    初次等待秒数
    :param cache_path:    若提供路径，则会使用该文件来缓存完成的 (env, text) => completion_array
    :return: (confusion_matrix, predicted_sequences, tasks, text_list)
        - confusion_matrix: shape=(N,N)，matrix[i,j] = [0..1]，最后帧
        - predicted_sequences: list(N)，i==j 的整段序列(均 [0..1])
        - tasks:    行标签 (环境列表)
        - text_list 列标签 (文本列表)
    """

    # 1) 读取任务列表
    task_subset = json.load(open(json_path, 'r'))
    if set_type == "train":
        tasks = task_subset["training_tasks"]
    elif set_type == "eval":
        tasks = task_subset["eval_tasks"]
    else:
        tasks = task_subset["test_tasks"]
    num_tasks = len(tasks)

    # ============ 读/写缓存 ============

    # 初始化缓存字典: key: (env_name, text_str) => value: completion_array
    cache_dict = {}
    if cache_path and os.path.exists(cache_path):
        print(f"[Info] Loading cache from {cache_path} ...")
        with open(cache_path, 'rb') as cf:
            cache_dict = pickle.load(cf)
        print(f"[Info] Loaded {len(cache_dict)} items from cache.")

    def save_cache():
        """将 cache_dict 写回到 cache_path (若指定)"""
        if cache_path:
            with open(cache_path, 'wb') as cf:
                pickle.dump(cache_dict, cf)

    # ============ 打开 HDF5 ============

    with h5py.File(h5_path, 'r') as f:
        # NxN 混淆矩阵
        confusion_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        # i==j 时的整段序列
        predicted_sequences: List[Optional[List[float]]] = [None]*num_tasks

        # 读取所有文本 (对应列 j)
        text_list: List[str] = []
        for env_name in tasks:
            text_dset_name = f"text_annotations/{env_name}_text"
            text_bytes = f[text_dset_name][()]
            text_str = text_bytes.decode("utf-8")
            text_list.append(text_str)

        # 3) 遍历行(环境 i)
        for i, env_video_name in enumerate(tasks):
            frames_data = f[env_video_name][:]  # (F, H, W, 3)

            # 4) 遍历列(文本 j)
            for j in range(num_tasks):
                text_str = text_list[j]
                if i == j:
                    # ============ 检查缓存 ============
                    cache_key = (env_video_name, text_str)
                    if cache_key in cache_dict:
                        # 已有缓存
                        completion_array = cache_dict[cache_key]
                        print(f"[Info] Using cache for (env={env_video_name}, text={text_str}).")
                        print(completion_array)
                    else:
                        # 需要请求
                        completion_array = []
                        for attempt in range(max_retries):
                            try:
                                analyzer = analyzer_class(
                                    api_key=gemini_api_key,
                                    frames_array=frames_data,
                                    task_description=text_str,
                                    max_frames=max_frames,
                                    offset=offset
                                )
                                completion_array = analyzer.run_analysis()  # => [0..100]
                                break  # 成功 => break
                            except requests.exceptions.HTTPError as e:
                                if e.response is not None and e.response.status_code == 429:
                                    wait_time = base_delay * (2 ** attempt)
                                    print(f"[Warning] 429 Too Many Requests - Retry {attempt+1}/{max_retries} after {wait_time:.1f}s...")
                                    time.sleep(wait_time)
                                else:
                                    raise e
                        else:
                            # 超过 max_retries
                            raise RuntimeError(f"[Error] 429: Max retries exceeded for (env={env_video_name}, text={text_str}).")

                        # 拿到 completion_array => 存入 cache 并保存
                        cache_dict[cache_key] = completion_array
                        save_cache()  # 立即写回

                    # 归一化到 [0..1]
                    completion_array = [0.0 if val is None else val/100.0 for val in completion_array]

                    # -- 混淆矩阵 只取最后帧
                    if len(completion_array) == 0:
                        last_value = 0.0
                    else:
                        last_value = completion_array[-1]
                    confusion_matrix[i, j] = last_value

                    # -- 如果 i==j，存整段序列
                    if i == j:
                        predicted_sequences[i] = completion_array

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
    # ============ 1) 生成 NxN 预测矩阵 =============
    conf_mat, all_seqs, tasks, text_list = generate_gemini_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/metaworld_GT_eval_v2.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        gemini_api_key="",
        analyzer_class=GeminiVideoAnalyzerHDF5,  # 你的类
        max_frames=15,
        offset=0.5,
        cache_path="gemini_gvl_cache_updated.pkl"
    )
    print(conf_mat.shape)  # NxN
    # print(all_seqs)     # N

    wandb.init(project="roboclip-v2", name=f"eval_GVL")

    # ============ 2) 计算相关(只用对角线) ============
    compute_pearson_correlation_from_sequences(
        all_seqs=all_seqs,
        set_type="eval",
        project_name="roboclip-v2",
        env_names=tasks
    )

    # ============ 3) 绘制混淆矩阵 =============
    plot_confusion_matrix_from_predictions(
        predicted_rewards=conf_mat,
        task_names=tasks,
        set_type="eval",
        text_instructions=text_list
    )


    # ============ 4) 计算 MSE ============
    compute_mse_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval"
    )

    # ============ 5) 计算 Spearman 相关系数 ============
    compute_spearman_correlation_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval"
    )

    conf_mat_close_success, _, _, _ = generate_gemini_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/close_succ_videos.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        gemini_api_key="",
        analyzer_class=GeminiVideoAnalyzerHDF5,  # 你的类
        max_frames=15,
        offset=0.5,
        cache_path="gemini_gvl_cache_close_success.pkl"
    )

    conf_mat_all_fail, _, _, _ = generate_gemini_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/all_fail_videos.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        gemini_api_key="",
        analyzer_class=GeminiVideoAnalyzerHDF5,  # 你的类
        max_frames=15,
        offset=0.5,
        cache_path="gemini_gvl_cache_all_fail.pkl"
    )
    rank_comparison(conf_mat_all_fail, conf_mat_close_success, conf_mat)
    wandb.finish()