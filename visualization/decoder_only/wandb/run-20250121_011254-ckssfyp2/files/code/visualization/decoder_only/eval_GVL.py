import numpy as np
import json
import h5py
from typing import Dict, List, Union, Optional
from scipy.stats import pearsonr
import wandb
import matplotlib.pyplot as plt
from GVL import GeminiVideoAnalyzerHDF5
import requests
import time


def shorten_name(name, separator=" ", max_length=5):
    parts = name.split(separator)
    return separator.join([part[:max_length] for part in parts])

def plot_matrix_as_image(matrix, names, set, text):
    # Create a figure and axis
    # only keep 2 decimal points
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.1, len(matrix)))
    
    # Plot the matrix with a colormap (darker = higher values)
    # cax = ax.matshow(matrix, cmap='viridis', interpolation='nearest')

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    # Add color bar
    # plt.colorbar(cax)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Set x-axis and y-axis ticks
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    # shortened_text = [shorten_name(name, max_length = 6) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 6) for name in names]

    shortened_text = [shorten_name(name, max_length = 12) for name in text]
    shortened_names = [shorten_name(name, separator = "-", max_length = 12) for name in names]

    # Label each row and column with the given names
    ax.set_xticklabels(shortened_text, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(shortened_names, fontsize=14)

    # Display the values in the matrix
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > np.max(matrix)/2 else 'black',  fontsize=14)
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


def generate_gemini_data(
    h5_path: str,
    json_path: str,
    set_type: str,
    gemini_api_key: str,
    analyzer_class,
    max_frames: int = 15,
    offset: float = 0.5,
    max_retries: int = 5,
    base_delay: float = 2.0
):
    """
    同时生成:
      1) (N x N) 混淆矩阵: 对 (视频 env_i, 文本 env_j) 取推理结果的最后一帧 (归一化至[0..1])
      2) (N 条完整序列): 对每个环境 i，仅在 (env_i, text_i) 这对时，保存整段预测序列(做相关性分析, 归一化至[0..1])

    启用指数退避：当 run_analysis() 遇到 HTTP 429 时，等待 base_delay * 2^attempt 再重试。
    如果重试 max_retries 次后仍失败，**raise** 异常并中断。

    :param h5_path:        HDF5 文件路径
    :param json_path:      new_task_v2.json 路径, 内含 {"training_tasks": [...], "eval_tasks": [...], "test_tasks": [...]}
    :param set_type:       "train" or "eval" or "test"
    :param gemini_api_key: Gemini API Key
    :param analyzer_class: 类似 GeminiVideoAnalyzerHDF5, 拥有 run_analysis() 方法
    :param max_frames:     每段视频采样帧数上限
    :param offset:         采样帧时的偏移
    :param max_retries:    当遇到 429 时的最大重试次数
    :param base_delay:     初次等待秒数
    :return: (confusion_matrix, predicted_sequences, tasks, text_list)
        - confusion_matrix: shape=(N,N)，matrix[i,j] = 最后一帧完成度(0..1)
        - predicted_sequences: list of length N，predicted_sequences[i] = 对 (env_i, text_i) 的完整序列 (List[float], [0..1])
        - tasks: 任务名称列表 (行环境)
        - text_list: 文本描述列表 (列指令)
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

    # 2) 打开 HDF5 文件
    with h5py.File(h5_path, 'r') as f:
        # 准备 NxN 矩阵，用于混淆矩阵
        confusion_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        # 准备一个列表，用于存储每个环境在 "匹配文本" (i==j) 时的整段预测序列 (归一化到 [0..1])
        predicted_sequences: List[Optional[List[float]]] = [None]*num_tasks

        # 读取所有文本描述(对应列 j)
        text_list: List[str] = []
        for env_name in tasks:
            text_dset_name = f"text_annotations/{env_name}_text"
            text_bytes = f[text_dset_name][()]
            text_str = text_bytes.decode("utf-8")
            text_list.append(text_str)
        
        # 3) 遍历行(环境 i)
        for i, env_video_name in enumerate(tasks):
            frames_data = f[env_video_name][:]

            # 4) 遍历列(文本 j)
            for j in range(num_tasks):
                text_str = text_list[j]

                # ============== 指数退避 + run_analysis ==============
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
                        completion_array = analyzer.run_analysis()  # e.g. [0..100], length <= max_frames
                        # 如果成功执行到这里，说明无异常 => break
                        break
                    except requests.exceptions.HTTPError as e:
                        # 判断是否 429 Too Many Requests
                        if e.response is not None and e.response.status_code == 429:
                            wait_time = base_delay * (2 ** attempt)
                            print(f"[Warning] 429 Too Many Requests - Retry {attempt+1}/{max_retries} after {wait_time:.1f}s...")
                            time.sleep(wait_time)
                        else:
                            # 如果是其它错误，或 e.response 为 None => 不重试，直接抛出
                            raise e
                else:
                    # Python for-else结构: 如果 for 在没有 break 的情况下结束 => 重试失败
                    error_msg = (
                        f"[Error] 429: Max retries ({max_retries}) exceeded. "
                        f"Failed on (env={env_video_name}, text={text_str})."
                    )
                    raise RuntimeError(error_msg)

                # 走到这里说明成功拿到了 completion_array
                # 缩放到 [0..1]
                completion_array = [
                    0.0 if val is None else val/100.0
                    for val in completion_array
                ]

                # -- 4.1) 混淆矩阵: 只取最后帧
                if len(completion_array) == 0:
                    last_value = 0.0
                else:
                    last_value = completion_array[-1]  # [0..1]

                confusion_matrix[i, j] = last_value

                # -- 4.2) 如果 i == j => 保存整段序列
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
        wandb.log({f"{set_type}/{env_name}_pearson": corr_val})

    # 计算平均
    if len(correlations) > 0:
        avg_corr = float(np.mean(correlations))
    else:
        avg_corr = 0.0

    wandb.log({f"{set_type}/average_pearson": avg_corr})
    wandb.finish()

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



    
if __name__ == "__main__":
    # ============ 1) 生成 NxN 预测矩阵 =============
    conf_mat, all_seqs, tasks, text_list = generate_gemini_data(
        h5_path="metaworld_GT_eval_v2.h5",
        json_path="new_task_v2.json",
        set_type="test",
        gemini_api_key="AIzaSyCaDj-o-VuadUwA94U9VdirB81VsY_t3TM",
        analyzer_class=GeminiVideoAnalyzerHDF5,  # 你的类
        max_frames=15,
        offset=0.5,
    )
    print(conf_mat.shape)  # NxN
    print(all_seqs)     # N

    wandb.init(project="roboclip-v2", name=f"test_GVL")

    # ============ 2) 计算相关(只用对角线) ============
    compute_pearson_correlation_from_sequences(
        all_seqs=all_seqs,
        set_type="test",
        project_name="roboclip-v2",
        env_names=tasks
    )

    # ============ 3) 绘制混淆矩阵 =============
    plot_confusion_matrix_from_predictions(
        predicted_rewards=conf_mat,
        task_names=tasks,
        set_type="test",
        text_instructions=text_list
    )