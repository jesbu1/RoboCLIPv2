import numpy as np
import json
import h5py
from typing import Dict, List, Union, Optional
from scipy.stats import pearsonr
import wandb
import matplotlib.pyplot as plt
from GVL import GeminiVideoAnalyzerHDF5


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
    offset: float = 0.5
):
    """
    同时生成:
      1) (N x N) 混淆矩阵: 对 (视频 env_i, 文本 env_j) 取推理结果的最后一帧
      2) (N 条完整序列): 对每个环境 i，仅在 (env_i, text_i) 这对时，保存整段帧序列(做相关性分析)

    :param h5_path:       HDF5 文件路径
    :param json_path:     new_task_v2.json 路径, 内含 { "training_tasks": [...], "eval_tasks": [...] }
    :param set_type:      "train" 或 "eval"
    :param gemini_api_key:Gemini API Key
    :param analyzer_class:类似 GeminiVideoAnalyzerHDF5, 拥有 run_analysis() 方法
    :param max_frames:    每段视频采样帧数上限
    :param offset:        采样帧时的偏移
    :return: (confusion_matrix, predicted_sequences, tasks, text_list)
        - confusion_matrix: shape = (N,N), matrix[i,j] = 最后一帧完成度(0~100)
        - predicted_sequences: list of length N, predicted_sequences[i] = 对 (env_i, text_i) 的完整序列 (List[float])
        - tasks: 任务名称列表 (行环境)
        - text_list: 文本描述列表 (列指令)
    """

    # 1) 载入任务列表
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

        # 准备一个列表，用于存储每个环境在"匹配文本"(即 i==j)时的整段预测序列
        predicted_sequences: List[Optional[List[float]]] = [None]*num_tasks

        # 先读取所有文本描述(对应列 j)
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

                # 调用 Gemini
                analyzer = analyzer_class(
                    api_key=gemini_api_key,
                    frames_array=frames_data,
                    task_description=text_str,
                    max_frames=max_frames,
                    offset=offset
                )
                completion_array = analyzer.run_analysis()  # e.g. [0..100], length <= max_frames
                
                # -- 4.1) 混淆矩阵: 只取最后帧
                if len(completion_array) == 0:
                    last_value = 0.0
                else:
                    last = completion_array[-1]
                    last_value = last if last is not None else 0.0
                confusion_matrix[i, j] = last_value

                # -- 4.2) 如果 i==j，说明这是“匹配环境+文本”，则保存整段序列
                #         (做帧级相关性或其它逐帧分析)
                if i == j:
                    # 替换 None 为 0.0
                    if completion_array:
                        full_seq = [val if val is not None else 0.0 for val in completion_array]
                    else:
                        full_seq = []
                    predicted_sequences[i] = full_seq
        
        # 返回4项: 混淆矩阵、整段序列列表、任务名称列表、文本列表
        return confusion_matrix, predicted_sequences, tasks, text_list



def compute_pearson_correlation_from_predictions(
    predicted_rewards: np.ndarray,
    set_type: str,
    project_name: str = "pearson_corr_project"
):
    """
    只负责对 NxN 的 predicted_rewards 做相关性计算。
    例如：只对矩阵的对角线 elements => /100 => 与 linspace(1..N) 做 pearson。
    
    不调用 Gemini，不读取 HDF5，只消费 predicted_rewards。
    """
    # 假设 NxN
    n, m = predicted_rewards.shape
    if n != m:
        print("[!] predicted_rewards 不是 NxN，当前函数只对 NxN 做演示！")
        return

    # wandb init
    wandb.init(project=project_name, name=f"{set_type}_pearson_corr")

    # 取对角线 => => [0..100]
    diagonal_values = np.diagonal(predicted_rewards)  # shape (N,)
    # 除以 100 => [0..1]
    pred_array = diagonal_values / 100.0
    # 构造参考 [1..N]
    gt_array = np.linspace(1, n, n, dtype=np.float32)

    if n < 2:
        pearson_val = 0.0
    else:
        pearson_val, _ = pearsonr(pred_array, gt_array)

    print(f"Pearson correlation (diag) = {pearson_val:.4f}")
    wandb.log({f"{set_type}/pearson": pearson_val})
    wandb.finish()




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

    wandb.init(project="roboclip-v2", name=f"{set_type}_confusion_matrix")
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
    print(len(all_seqs))  # N
    print(all_seqs)     # N
    # tasks 列表也要读出来，以便后面画图
    with open("new_task_v2.json") as f:
        subset_info = json.load(f)
    train_tasks = subset_info["test_tasks"]  # or eval_tasks

    # ============ 2) 计算相关(只用对角线) ============
    # compute_pearson_correlation_from_predictions(
    #     predicted_rewards=rewards,
    #     set_type="eval",
    #     project_name="roboclip-v2"
    # )

    # ============ 3) 绘制混淆矩阵 =============
    plot_confusion_matrix_from_predictions(
        predicted_rewards=conf_mat,
        task_names=train_tasks,
        set_type="test",
        text_instructions=text_list
    )