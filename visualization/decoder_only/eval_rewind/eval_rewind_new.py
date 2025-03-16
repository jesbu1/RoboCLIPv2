import torch
import torch.nn.functional as F
import numpy as np
import wandb
import h5py
import os
import json
import pickle
import matplotlib.pyplot as plt
import textwrap
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# from models import ClassProgressTransformer
from model_pe import ClassProgressTransformer

from PIL import Image
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModel
import io
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)

DINO_BATCH_SIZE = 128
MAX_NUM_FRAMES_PER_EPISODE = 128


def animate_incremental(frames_tensor, incremental_rewards, fps=15):
    """
    Create an animation that shows frames on the left and the incremental reward curve on the right.

    Args:
        frames_tensor: torch.Tensor of shape [N, C, H, W], pixel range [0,1].
        incremental_rewards: list or array of length N, each element is a reward.
        fps: GIF framerate.

    Returns:
        gif_buffer: in-memory BytesIO containing the GIF.
    """
    # 1) Convert frames to numpy for display
    # if isinstance(frames_tensor, torch.Tensor):
    #     frames_np = frames_tensor.cpu().numpy()
    # else:
    #     frames_np = frames_tensor
    # frames_np = (frames_np * 255).astype(np.uint8)
    # frames_np = np.transpose(frames_np, (0, 2, 3, 1))

    # 2) Prepare figure
    # n = len(frames_np)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # image_plot = ax1.imshow(frames_np[0])
    ax1.set_title('Video Frames')
    ax1.axis('off')

    # ax2.set_title('Incremental Rewards')
    ax2.set_xlim(0, len(incremental_rewards) - 1)
    y_min, y_max = min(incremental_rewards), max(incremental_rewards)
    y_range = y_max - y_min if y_max != y_min else 1
    # ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax2.set_ylim(-1,1)

    line_plot, = ax2.plot([], [], lw=2, color='blue')
    scat = ax2.scatter([], [], color='red', zorder=5)

    gif_buffer = io.BytesIO()
    images = []

    # 3) Update frames one by one
    for frame_idx in range(len(incremental_rewards)):
        # image_plot.set_array(frames_np[frame_idx])
        line_plot.set_data(np.arange(frame_idx + 1), incremental_rewards[:frame_idx + 1])
        scat.set_offsets(np.array([[frame_idx, incremental_rewards[frame_idx]]]))
        fig.canvas.draw()

        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = img_array.reshape((h, w, 3))
        images.append(Image.fromarray(img_array))

    # 4) Save to GIF
    images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
    gif_buffer.seek(0)

    # ---------- 新增：保存 ax2 (Incremental Rewards) 子图为单独 PNG ----------
    # 先确保最后一个帧(也就是最终曲线)已经绘制完
    # （其实上面for循环已画完，所以这里直接获取ax2 bbox即可）
    # 注意：有时文字或ticks会被裁得太紧，可以通过 expanded() 留些余地
    ax2.tick_params(labelleft=False, left=False)  # 隐藏左侧坐标轴

    # 1) 获取 ax2 相对于整个 figure 的边界（单位：像素）
    extent_px = ax2.get_window_extent()

    # 2) 将像素坐标转换为英寸坐标 (fig.dpi_scale_trans)
    extent_in = extent_px.transformed(fig.dpi_scale_trans.inverted())
    
    # 提取宽、高，并计算中心
    width = extent_in.width
    height = extent_in.height
    center_x = extent_in.x0 + width / 2.0
    center_y = extent_in.y0 + height / 2.0

    # 3) 取最大边作为正方形的边长
    side = max(width, height)

    # 4) 以 (center_x, center_y) 为中心，构建新的正方形 bbox
    x0 = center_x - side / 2.0
    x1 = center_x + side / 2.0
    y0 = center_y - side / 2.0
    y1 = center_y + side / 2.0

    # 先构建不放大的正方形 bbox
    square_extent = Bbox.from_extents(x0, y0, x1, y1)
    
    fig.savefig(
        "ax2_plot_rewind.pdf",         # 你想要保存的文件名
        bbox_inches=square_extent.expanded(1.2, 1.2)  # 适当放大 1.2 倍，以免标签被裁切
    )
    # ---------------------------------------------------------------------


    plt.close(fig)
    return gif_buffer


def rank_comparison(cm1, cm2, cm3):
    """
    只比较混淆矩阵对角线上的值 (每个任务的正确预测数)，
    给出每个矩阵在各任务上的排名(1=最好,3=最差)，
    并统计每个矩阵拿到排名1,2,3的概率，最后上传到W&B。

    另外增加一个“GT排名”指标，要求对角线满足 cm1 < cm2 < cm3 的任务数占比(成功率)。

    新增：
    - 计算与 GT=[3,2,1] 的 Spearman 相关系数，并对所有任务取平均。
    """

    # 1. 取出三个矩阵的对角线
    diag1 = np.diag(cm1)
    diag2 = np.diag(cm2)
    diag3 = np.diag(cm3)

    num_tasks = len(diag1)

    # ranks[i, j] 表示第 i 个任务下，第 j 个矩阵的排名(1/2/3)
    ranks = np.zeros((num_tasks, 3), dtype=np.int32)

    # 2. 逐任务比较并排名
    gt_count = 0  # 统计满足 cm1 < cm2 < cm3 的次数

    for i in range(num_tasks):
        d1, d2, d3 = diag1[i], diag2[i], diag3[i]
        values = [d1, d2, d3]
        
        # 从小到大排序，返回索引
        sorted_indices = np.argsort(values)  # ascending order
        
        # sorted_indices[0] -> 最小值, sorted_indices[2] -> 最大值
        # 我们要给 最大值 rank=1, 次大 rank=2, 最小 rank=3
        ranks[i, sorted_indices[0]] = 3  # 最小值 => rank 3
        ranks[i, sorted_indices[1]] = 2
        ranks[i, sorted_indices[2]] = 1  # 最大值 => rank 1

        # 检查“GT 排名”条件： cm1 < cm2 < cm3
        if d1 <= d2 < d3:
            gt_count += 1

    # 3. 计算平均排名 (axis=0 => 对每列求平均)
    avg_ranks = np.mean(ranks, axis=0)  # 形状 (3,)

    # 4. 统计每个矩阵在 num_tasks 个任务中 rank=1,2,3 的次数
    rank1_counts = np.sum(ranks == 1, axis=0)  # (3,)
    rank2_counts = np.sum(ranks == 2, axis=0)
    rank3_counts = np.sum(ranks == 3, axis=0)

    # 转换为概率(出现次数 / 任务数)
    rank1_probs = rank1_counts / num_tasks
    rank2_probs = rank2_counts / num_tasks
    rank3_probs = rank3_counts / num_tasks

    # 5. 计算“GT 排名”成功率
    gt_success_rate = gt_count / num_tasks if num_tasks > 0 else 0.0

    # ========== 新增：计算 Spearman 相关系数(与 GT=[3,2,1])并对所有任务取平均 ==========
    # ground truth 排名
    gt_ranks = np.array([3, 2, 1])

    spearman_sum = 0.0
    # 对每个任务的 [cm1_rank, cm2_rank, cm3_rank] 计算与 [3,2,1] 的 Spearman 相关系数
    for i in range(num_tasks):
        predicted = ranks[i, :]  # e.g. [3,2,1] / [2,1,3] etc.
        # 计算
        rho, p_value = spearmanr(gt_ranks, predicted)
        spearman_sum += rho

    # 取平均
    spearman_avg = spearman_sum / num_tasks if num_tasks > 0 else 0.0

    # 6. 打印结果
    print("cm1 对角线:", diag1)
    print("cm2 对角线:", diag2)
    print("cm3 对角线:", diag3)
    print("\nranks (每行对应第 i 个任务下 [cm1, cm2, cm3] 的排名):\n", ranks)
    print("\n平均排名 (数值越小表示越好):", avg_ranks)
    print("rank=1 次数:", rank1_counts, " 概率:", rank1_probs)
    print("rank=2 次数:", rank2_counts, " 概率:", rank2_probs)
    print("rank=3 次数:", rank3_counts, " 概率:", rank3_probs)
    print("\nGT 排名成功率 (cm1 < cm2 < cm3):", gt_success_rate)
    print("\n平均 Spearman 相关系数 (对所有任务与 [3,2,1] 的比较):", spearman_avg)

    # 7. 上传到W&B
    wandb.log({
        # 平均排名
        "Average Rank/cm1": float(avg_ranks[0]),
        "Average Rank/cm2": float(avg_ranks[1]),
        "Average Rank/cm3": float(avg_ranks[2]),

        # rank=1 概率
        "Rank1 Prob/cm1": float(rank1_probs[0]),
        "Rank1 Prob/cm2": float(rank1_probs[1]),
        "Rank1 Prob/cm3": float(rank1_probs[2]),

        # rank=2 概率
        "Rank2 Prob/cm1": float(rank2_probs[0]),
        "Rank2 Prob/cm2": float(rank2_probs[1]),
        "Rank2 Prob/cm3": float(rank2_probs[2]),

        # rank=3 概率
        "Rank3 Prob/cm1": float(rank3_probs[0]),
        "Rank3 Prob/cm2": float(rank3_probs[1]),
        "Rank3 Prob/cm3": float(rank3_probs[2]),

        # 新增“GT 排名”成功率
        "GT Ranking Success Rate": float(gt_success_rate),

        # 新增 Spearman 相关系数(平均)
        "Spearman Average (GT=[3,2,1])": float(spearman_avg)
    })

    # 可根据需要 return 结果
    return avg_ranks, rank1_probs, rank2_probs, rank3_probs, gt_success_rate, spearman_avg


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def sample_embedding_frames(embeddings, num_frames = 32):
    total_frames = embeddings.shape[0]
    if total_frames > num_frames:
        index = np.linspace(0, total_frames-1, num_frames).astype(int)
        embeddings = embeddings[index]

    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        first_frame = embeddings[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_num, 1)
        embeddings = torch.cat([padding_frames, embeddings], dim=0)
    return embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=True)
dinov2_vits14 = dinov2_vits14.to(device)
dino_transform_image = T.Compose(
    [T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
)

def dino_load_image(img: np.ndarray) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.fromarray(img)

    transformed_img = dino_transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def padding_embedding(video_frames, max_length):
    video_length = len(video_frames)
    if type(video_frames) == np.ndarray:
        video_frames = torch.tensor(video_frames)
    if video_length < max_length:
        # padding first frame
        padding_length = max_length - video_length
        first_frame = video_frames[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_length, 1)
        video_frames = torch.cat([padding_frames, video_frames], dim=0)
    
    elif video_length > max_length:
        frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
        video_frames = video_frames[frame_idx]

    return video_frames


def sample_embedding_frames(embeddings, num_frames = 32):
    total_frames = embeddings.shape[0]
    if total_frames > num_frames:
        index = np.linspace(0, total_frames-1, num_frames).astype(int)
        embeddings = embeddings[index]

    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        first_frame = embeddings[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_num, 1)
        embeddings = torch.cat([padding_frames, embeddings], dim=0)
    return embeddings


def plot_matrix_as_image(matrix, names, set, text, fig_name):
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

    # shortened_text = [shorten_text(name, max_length = 25) for name in text]
    # shortened_names = [shorten_name(name, separator = "-", max_length = 12) for name in names]

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
    wandb.log({f"confusion_matrix/{fig_name}": wandb.Image(fig)})
    plt.savefig(f"confusion_matrix_{fig_name}.pdf", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


def compute_rewind_reward(rewind_model, args, episode_image_embeddings, lang_embeddings, gif_path="incremental_reward_rewind.gif"):
    One_step = args.
    reward_seq = []

    if args.normalize_embedding:
        lang_embeddings = normalize_embeddings(lang_embeddings)
        episode_image_embeddings = normalize_embeddings(episode_image_embeddings)
    for i in range(1, episode_image_embeddings.shape[0]+1):
        partial_image_embeddings = episode_image_embeddings[:i]
        if args.subsample_video:
            processed_video_embedding = sample_embedding_frames(
                partial_image_embeddings, args.max_length
            ).unsqueeze(0)

        pred_class, two_step_class = rewind_model(processed_video_embedding, lang_embeddings)
        if One_step:
            pred_class = pred_class 
            predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
            predicted_classes = predicted_classes[1:]  # Remove first frame prediction
        else:
            two_step_class_prob = (two_step_class.squeeze() > args.binary_threshold).float()
            pred_class = pred_class * two_step_class_prob

            # pred_class shape => [1, T] or [T], check your model output
            predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
            # user code removes first frame => predicted_classes[1:]
            predicted_classes = predicted_classes[1:]
        # print(predicted_classes)
        # final = last index
        video_reward = predicted_classes[-1] if len(predicted_classes) > 0 else 0.0
        reward_seq.append(video_reward)
        print(reward_seq)

    gif_buffer = animate_incremental(None, reward_seq, fps=15)
    with open(gif_path, 'wb') as f:
        f.write(gif_buffer.getvalue())
    print(f"Saved incremental reversed GIF to {gif_path}")
    # TODO: Wandb

    return reward_seq


def generate_rewind_data(
    h5_path: str,
    json_path: str,
    set_type: str,
    rewind_model: torch.nn.Module,
    device: str = "cuda",
    cache_path="rewind_cache.pkl",
    args=None,
    annotation = None,
    One_step = False
):
    """
    与 generate_gemini_data 类似，遍历 (环境, 文本) 组合，生成:
      1) (N x N) 混淆矩阵 (最后帧单值)
      2) (N 条完整序列, 但每条里包含5个示例的逐帧/增量结果)，只对 i==j 的组合做“逐帧”或“增量”推理

    返回:
        (confusion_matrix, predicted_sequences, tasks, text_list)
          confusion_matrix: shape=(N,N), float，表示 [0..1] reward (5个demo的平均)
          predicted_sequences: List[Optional[List[List[float]]]]
              # predicted_sequences[i] = [ seq_demo_0, seq_demo_1, ..., seq_demo_4 ]
              # 每个 seq_demo_k 是一段逐帧/增量结果
          tasks: 任务名称列表
          text_list: 文本说明列表 (在这里是 minilm_lang_embedding, shape=[N, 1, 384] 之类)
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
    # 2) 打开 HDF5
    with h5py.File(h5_path, "r") as f:
        
        # 为混淆矩阵 (N x N) 分配空间
        confusion_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        # predicted_sequences[i] => None 或者 [[seq_for_demo0], [seq_for_demo1], ... , [seq_for_demo4]]
        predicted_sequences = [None] * num_tasks

        # 3) 读取所有文本 (与 tasks 对应) => text_list
        text_list = []
        for env in tasks:
            # Read minilm_lang_embedding for each env
            group = f[env]
            if annotation != None:
                text_arr = np.asarray(group[f"minilm_lang_embedding_{annotation}"])  # shape (1, 384)
            else:
                text_arr = np.asarray(group["minilm_lang_embedding"])
            text_list.append(text_arr)
        # e.g. shape: (N, 1, 384) after stacking
        text_list = np.stack(text_list)
        text_list = torch.tensor(text_list).to(device).float()

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

        # 5) 遍历所有 (i, j)
        for i, env_video_name in enumerate(tasks):
            group = f[env_video_name]

            # Gather all 5 embeddings
            # Some HDF5 might not have exactly 5, so we do `range(5)` but check if dataset exists.
            # 1) 找到所有数字型 key, 排序后取前 5
            digit_keys = sorted([k for k in group.keys() if k.isdigit()], key=lambda x: int(x))
            selected_keys = digit_keys[:5]

            # 如果不等于 ["0","1","2","3","4"] 才打印
            if selected_keys != ["0", "1", "2", "3", "4"]:
                print(f"[Info] {env_video_name} selected demos: {selected_keys}")

            # 2) 收集它们对应的 embedding
            all_video_embeddings = []
            for key_name in selected_keys:
                embed_arr = np.asarray(group[key_name])  # shape (32, 768)
                video_emb = torch.tensor(embed_arr).to(device).float()
                all_video_embeddings.append(video_emb)

            if len(all_video_embeddings) == 0:
                # If no embeddings found, skip
                print(f"Warning: no embeddings for {env_video_name}")
                continue

            # For partial-sequence saving:
            # predicted_seq_for_5 = list of 5 partial sequences
            # We'll only fill it if i==j
            predicted_seq_for_5 = [[] for _ in range(len(all_video_embeddings))]

            # 5.2) 遍历所有文本 embedding j
            for j, text_embedding in enumerate(text_list):
                # Optional embedding normalization
                if args.normalize_embedding:
                    text_embedding = normalize_embeddings(text_embedding)

                # ---- A) 计算(5个demo)的final reward，并求平均 ----
                final_rewards = []
                reward_seq = []
                for demo_id, video_embedding in enumerate(all_video_embeddings):
                    # Use partial/final cache key => (env_video_name, j, demo_id, "full")
                    cache_key_full = (env_video_name, j, demo_id, "full")

                    if cache_key_full in cache_dict:
                        video_reward = cache_dict[cache_key_full]
                    else:
                        # 这里示例只是做 sample_embedding_frames => forward
                        if args.normalize_embedding:
                            video_embedding = normalize_embeddings(video_embedding)
                        if args.subsample_video:
                            processed_video_embedding = sample_embedding_frames(
                                video_embedding, args.max_length
                            ).unsqueeze(0)

                        # forward
                        pred_class, two_step_class = rewind_model(processed_video_embedding, text_embedding)
                        if One_step:
                            pred_class = pred_class 
                            predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
                            predicted_classes = predicted_classes[1:]  # Remove first frame prediction
                        else:
                            two_step_class_prob = (two_step_class.squeeze() > args.binary_threshold).float()
                            pred_class = pred_class * two_step_class_prob

                            # pred_class shape => [1, T] or [T], check your model output
                            predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
                            # user code removes first frame => predicted_classes[1:]
                            predicted_classes = predicted_classes[1:]

                        reward_seq.append(predicted_classes)
                        # final = last index
                        video_reward = predicted_classes[-1] if len(predicted_classes) > 0 else 0.0

                        cache_dict[cache_key_full] = video_reward
                        save_cache()

                    final_rewards.append(video_reward)

                # average final reward for confusion_matrix
                if len(final_rewards) > 0:
                    avg_reward = float(sum(final_rewards) / len(final_rewards))
                else:
                    avg_reward = 0.0

                confusion_matrix[i, j] = avg_reward

                # ---- B) 如果 i == j => 逐帧/增量 预测序列(对5条demo分别计算) ----
                if i == j:
                    for demo_id, video_embedding in enumerate(all_video_embeddings):
                        # Build partial seq: for partial_count in [1..args.max_length) 
                        partial_seq = []
                        # We can reuse the same "predicted_classes" from above or re-run.
                        # For simplicity, we show re-running in partial. But you could store them once above if you prefer.
                        # 
                        # Here we assume your model always expects full sequence,
                        # so we do sample_embedding_frames(video_embedding, partial_count).
                        # If your partial is truly partial frames, you do something else.
                        for partial_count in range(1, args.max_length):
                            cache_key_partial = (env_video_name, j, demo_id, partial_count)
                            if cache_key_partial in cache_dict:
                                sub_reward = cache_dict[cache_key_partial]
                            else:
                                
                                sub_reward = reward_seq[demo_id][partial_count-1] 

                                cache_dict[cache_key_partial] = sub_reward
                                save_cache()

                            partial_seq.append(sub_reward)

                        # store partial_seq
                        predicted_seq_for_5[demo_id] = partial_seq

                    # Now assign the 5 partial sequences to predicted_sequences[i]
                    predicted_sequences[i] = predicted_seq_for_5

        return confusion_matrix, predicted_sequences, tasks, text_list



def load_rewind_model(ckpt_path: str = "/scr/yusenluo/RoboCLIP/visualization/decoder_only/saved_models_final/Crop_MetaWorld_binary_thrd_0.5_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_20_lr_0.0001_progress_loss_weight_2.0_weighted_mse/epoch_15.pth") -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dim = 768
    text_dim = 384



    # model_path = "/scr/yusenluo/RoboCLIP/visualization/decoder_only/saved_models_final/Crop_MetaWorld_binary_thrd_0.5_Rewind_ratio_0.5_MiniLM_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_20_lr_0.0001_progress_loss_weight_2.0_weighted_mse/epoch_15.pth"
    model_dict = torch.load(ckpt_path)
    args = model_dict['args']


    self_attention_model = ClassProgressTransformer(
        args=args,
        video_dim=video_dim,  # Original video embedding dimension
        text_dim=text_dim,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)
    self_attention_model.load_state_dict(model_dict['model'])

    print("✅ Model loaded successfully!")
    return self_attention_model, args


def compute_pearson_correlation_from_sequences(
    all_seqs,
    env_names,
    set_type: str,
    project_name: str = "roboclip-v2"
):
    """
    现在假设:
      all_seqs[i] => 一个长度为5的列表, 
                     其中每个元素 seq_demo_k => 形如 [val0, val1, ..., valM].
      env_names[i] => 第 i 个环境名称 (str).

    对每个环境 i:
      1) 遍历它的 5 条序列, 计算每条序列的 Pearson 相关系数
         - 若序列全为0 或者长度<2, 则相关系数=0
      2) 求 5 条序列的相关系数平均值、标准差 -> wandb.log()
      3) 将该平均值保存到 env_avg_correlations

    最后再对 env_avg_correlations 做平均 (overall_avg)，也上传到 wandb。
    """

    if len(all_seqs) != len(env_names):
        print("[!] all_seqs 和 env_names 长度不一致，无法一一对应。")
        return 0.0, []

    env_avg_correlations = []  # 将每个环境的“平均相关系数”放进来

    for i, five_seqs in enumerate(all_seqs):
        env_name = env_names[i]

        # five_seqs = [seq_demo0, seq_demo1, seq_demo2, seq_demo3, seq_demo4]
        # 每个 seq_demo 是一条逐帧预测 (e.g. [10, 20, 30, ...])
        cor_vals = []

        for seq_demo in five_seqs:
            if len(seq_demo) < 2:
                # 若长度<2，无法算相关系数 => 0
                cor_vals.append(0.0)
                continue

            pred_array = np.array(seq_demo, dtype=np.float32)

            # 若全部为 0，则相关系数强行设为 0
            if np.allclose(pred_array, 0):
                cor_vals.append(0.0)
                continue

            n = len(pred_array)
            # 构造 [1..n] 作为“理想逐帧递增”参考
            gt_array = np.linspace(0, 1, n, dtype=np.float32)

            pearson, _ = pearsonr(pred_array, gt_array)
            cor_vals.append(float(pearson))

        # 计算这 5 个 demo 的平均 & 标准差
        if len(cor_vals) > 0:
            env_avg = float(np.mean(cor_vals))
            env_std = float(np.std(cor_vals))
        else:
            env_avg = 0.0
            env_std = 0.0

        # 逐环境 log
        wandb.log({
            f"{set_type}_pearson_correlation/{env_name}_avg": env_avg,
            f"{set_type}_pearson_correlation/{env_name}_std": env_std
        })

        env_avg_correlations.append(env_avg)

    # 对所有环境的平均
    if len(env_avg_correlations) > 0:
        overall_avg = float(np.mean(env_avg_correlations))
    else:
        overall_avg = 0.0

    # log overall
    wandb.log({f"{set_type}_pearson_correlation/overall_avg": overall_avg})

    print(f"[{set_type}] 所有环境的平均Pearson相关系数: {overall_avg:.4f}")

    # 返回 (overall_avg, 所有环境的平均值列表)
    return overall_avg, env_avg_correlations


def plot_confusion_matrix_from_predictions(
    predicted_rewards: np.ndarray,
    task_names: list,
    text_instructions: list,
    set_type: str,
    fig_name: str
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
        text=text_instructions,
        fig_name=fig_name
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
    project_name: str = "roboclip-v2"
):
    """
    与上述 pearson 函数类似, 这里用 spearmanr().
    all_seqs[i] => [5 条序列], 每条序列 => [val0, val1, ...]
    """

    if len(all_seqs) != len(env_names):
        print("[!] all_seqs 和 env_names 长度不一致，无法一一对应。")
        return 0.0, []

    env_avg_correlations = []

    for i, five_seqs in enumerate(all_seqs):
        env_name = env_names[i]
        cor_vals = []
        
        for j, seq_demo in enumerate(five_seqs):
            if len(seq_demo) < 2:
                cor_vals.append(0.0)
                continue

            pred_array = np.array(seq_demo, dtype=np.float32)
            # print(f"🔹 {env_name}_{j}", pred_array)
            if np.allclose(pred_array, 0):
                cor_vals.append(0.0)
                continue

            n = len(pred_array)
            gt_array = np.linspace(1, n, n, dtype=np.float32)

            spearman, _ = spearmanr(pred_array, gt_array)
            cor_vals.append(float(spearman))

        # 5 条序列的平均 & 标准差
        if len(cor_vals) > 0:
            env_avg = float(np.mean(cor_vals))
            env_std = float(np.std(cor_vals))
        else:
            env_avg = 0.0
            env_std = 0.0

        wandb.log({
            f"{set_type}_spearman_correlation/{env_name}_avg": env_avg,
            f"{set_type}_spearman_correlation/{env_name}_std": env_std
        })
        env_avg_correlations.append(env_avg)

    if len(env_avg_correlations) > 0:
        overall_avg = float(np.mean(env_avg_correlations))
    else:
        overall_avg = 0.0

    wandb.log({f"{set_type}_spearman_correlation/overall_avg": overall_avg})
    print(f"[{set_type}] 所有环境的平均Spearman相关系数: {overall_avg:.4f}")

    return overall_avg, env_avg_correlations


def load_rewind_model_progress(ckpt_path: str = "/scr/yusenluo/RoboCLIP/visualization/decoder_only/saved_models_final/Crop_MetaWorld_binary_thrd_0.5_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_20_lr_0.0001_progress_loss_weight_2.0_weighted_mse/epoch_15.pth") -> torch.nn.Module:
    path = os.path.join("saved_models", 
                    "ProgressOnlyCrop_MetaWorld_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_50_lr_0.0001_progress_loss_weight_1",
                    "epoch_35.pth")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dim = 768
    text_dim = 384

    model_dict = torch.load(ckpt_path)
    args = model_dict['args']
    model = ClassProgressTransformer(
            args=args,
            video_dim=video_dim,  # Original video embedding dimension
            text_dim=text_dim,   # Original text embedding dimension
            hidden_dim=512  # Common dimension for transformer processing
        ).to(device)
    # model = RewardOneStepNewPositionEmbeddingPredictor(args=args,
    #         video_dim=video_dim,  # Original video embedding dimension
    #         text_dim=text_dim,   # Original text embedding dimension
    #         hidden_dim=512  # Common dimension for transformer processing)
    # ).to(device)
    model.load_state_dict(model_dict['model'])
    model.eval()
    return model, args

def compute_spearman_correlation_multi_annotations(
    all_seqs_a,  # 第一个all_seqs, shape=[num_tasks, 5, variable_length]
    all_seqs_b,  # 第二个all_seqs
    all_seqs_c,  # 第三个all_seqs
    all_seqs_d,  # 第四个all_seqs
    env_names,   # 长度和 all_seqs_x 一致
    set_type: str,
):
    """
    对于每个task i:
      1) 从 all_seqs_a[i] 中取出5条序列 => 分别算spearman相关系数 => 平均 => corr_a
      2) 从 all_seqs_b[i] 中取出5条序列 => 分别算spearman相关系数 => 平均 => corr_b
      3) 从 all_seqs_c[i] 中取出5条序列 => 分别算spearman相关系数 => 平均 => corr_c
      4) 从 all_seqs_d[i] 中取出5条序列 => 分别算spearman相关系数 => 平均 => corr_d

      然后对 [corr_a, corr_b, corr_c, corr_d] 求平均 => task_avg_corr
                             以及 求方差 => task_avg_var (样本方差, ddof=1).

      最终对每个task的 (avg_corr, avg_var) 做全局平均 => overall_avg_corr, overall_avg_var
    """

    num_tasks = len(env_names)

    # 检查四个 all_seqs 长度是否一致
    if not (len(all_seqs_a) == len(all_seqs_b) == len(all_seqs_c) == len(all_seqs_d) == num_tasks):
        print("[Error] 四个all_seqs长度和env_names不一致")
        return 0.0, 0.0

    # 存放每个task的(平均相关系数, 方差)
    task_corrs = []
    task_vars = []

    for i in range(num_tasks):
        env_name = env_names[i]

        # --- 对 all_seqs_a[i] ---
        corr_a = compute_avg_spearman(all_seqs_a[i])
        corr_b = compute_avg_spearman(all_seqs_b[i])
        corr_c = compute_avg_spearman(all_seqs_c[i])
        corr_d = compute_avg_spearman(all_seqs_d[i])

        # 这四个平均相关系数 => [corr_a, corr_b, corr_c, corr_d]
        approach_corrs = np.array([corr_a, corr_b, corr_c, corr_d], dtype=np.float32)

        # 每个task的平均
        task_avg_corr = float(approach_corrs.mean())
        # 样本方差 (ddof=1 => 分母 = n-1 = 3)
        task_avg_var = float(np.var(approach_corrs, ddof=1))

        # 将本task的avg_corr和avg_var记录到wandb
        wandb.log({
            f"{set_type}_spearman_correlation/{env_name}_avg_corr": task_avg_corr,
            f"{set_type}_spearman_correlation/{env_name}_avg_var": task_avg_var
        })

        task_corrs.append(task_avg_corr)
        task_vars.append(task_avg_var)

    # 最后对所有task的 avg_corr 求平均
    if len(task_corrs) > 0:
        overall_avg_corr = float(np.mean(task_corrs))
        overall_avg_var  = float(np.mean(task_vars))
    else:
        overall_avg_corr = 0.0
        overall_avg_var  = 0.0

    # 把整体平均也上传wandb
    wandb.log({
        f"{set_type}_spearman_correlation/overall_avg_corr": overall_avg_corr,
        f"{set_type}_spearman_correlation/overall_avg_var": overall_avg_var
    })

    print(f"[{set_type}] 所有环境的所有instruction得平均Spearman相关系数: {overall_avg_corr:.4f}")
    print(f"[{set_type}] 所有环境的所有instruction平均方差: {overall_avg_var:.4f}")

    return overall_avg_corr, overall_avg_var


def compute_avg_spearman(five_seqs):
    """
    给定 5 条序列 => 各自算 Spearman 相关系数 => 取平均值
    (若seq长度<2或全0，则记为0)
    """
    cor_vals = []
    for seq in five_seqs:
        if len(seq) < 2:
            cor_vals.append(0.0)
            continue

        arr = np.array(seq, dtype=np.float32)
        if np.allclose(arr, 0):
            cor_vals.append(0.0)
            continue

        n = len(arr)
        gt = np.linspace(1, n, n, dtype=np.float32)
        r, _ = spearmanr(arr, gt)
        if np.isnan(r):
            r = 0.0
        cor_vals.append(float(r))

    if len(cor_vals) == 0:
        return 0.0
    return float(np.mean(cor_vals))


def generate_rewind_gif(
    h5_path: str,
    json_path: str,
    set_type: str,
    rewind_model: torch.nn.Module,
    device: str = "cuda",
    args=None,
    annotation = None,
):

    # 1) 从 new_task_v2.json 中读取 tasks
    task_subset = json.load(open(json_path, "r"))
    if set_type == "train":
        tasks = task_subset["training_tasks"]
    elif set_type == "eval":
        tasks = task_subset["eval_tasks"]
    else:
        tasks = task_subset["test_tasks"]
    num_tasks = len(tasks)

    # 2) 打开 HDF5
    with h5py.File(h5_path, "r") as f:

        # 3) 读取所有文本 (与 tasks 对应) => text_list
        text_list = []
        for env in tasks:
            # Read minilm_lang_embedding for each env
            group = f[env]
            if annotation != None:
                text_arr = np.asarray(group[f"minilm_lang_embedding_{annotation}"])  # shape (1, 384)
            else:
                text_arr = np.asarray(group["minilm_lang_embedding"])
            text_list.append(text_arr)
        # e.g. shape: (N, 1, 384) after stacking
        text_list = np.stack(text_list)
        text_list = torch.tensor(text_list).to(device).float()

        # 5) 遍历所有 (i, j)
        for i, env_video_name in enumerate(tasks):
            group = f[env_video_name]

            # Gather all 5 embeddings
            # Some HDF5 might not have exactly 5, so we do `range(5)` but check if dataset exists.
            # 1) 找到所有数字型 key, 排序后取前 5
            digit_keys = sorted([k for k in group.keys() if k.isdigit()], key=lambda x: int(x))
            selected_keys = digit_keys[:5]

            # 如果不等于 ["0","1","2","3","4"] 才打印
            if selected_keys != ["0", "1", "2", "3", "4"]:
                print(f"[Info] {env_video_name} selected demos: {selected_keys}")

            # 2) 收集它们对应的 embedding
            all_video_embeddings = []
            for key_name in selected_keys:
                embed_arr = np.asarray(group[key_name])  # shape (32, 768)
                video_emb = torch.tensor(embed_arr).to(device).float()
                all_video_embeddings.append(video_emb)

            if len(all_video_embeddings) == 0:
                # If no embeddings found, skip
                print(f"Warning: no embeddings for {env_video_name}")
                continue

            # 5.2) 遍历所有文本 embedding j
            for j, text_embedding in enumerate(text_list):
                if i != j:
                    continue
                for demo_id, video_embedding in enumerate(all_video_embeddings):
                    compute_rewind_reward(rewind_model=rewind_model, args=args, episode_image_embeddings=video_embedding, lang_embeddings=text_embedding, gif_path=f"rewind_gif/{env_video_name}_{demo_id}.gif")

if __name__ == "__main__":
    
    # rewind_model, model_args = load_rewind_model(ckpt_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/saved_models_final/Crop_MetaWorld_binary_thrd_0.5_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_20_lr_0.0001_progress_loss_weight_2.0_weighted_mse/epoch_15.pth")
    rewind_model, model_args = load_rewind_model(ckpt_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/NewPE_Crop_MetaWorld_binary_thrd_0.5_Rewind_ratio_0.5_MiniLM_AddOpenXData_ReWind_SubVideo_MaxLen16_PosEmb_CosScheduler_ClipGrad_View_side_ExtraDataRatio_0.2_epochs_20_lr_0.0001_progress_loss_weight_2.0/epoch_14.pth")
    print(model_args)
    # h5_eval_file = h5py.File("/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_embedding_1_demo_dataset_v2.h5", "r")
     # ============ 1) 生成 NxN 预测矩阵 =============
    confusion_matrix, all_seqs, tasks, text_list = generate_rewind_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        cache_path="final_rewind_cache_oxe_pos_end.pkl",
        args = model_args,
        One_step=args.
    )

    confusion_matrix_1, all_seqs1, _, _ = generate_rewind_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        cache_path="final_rewind_cache_oxe_pos_end_1.pkl",
        args = model_args,
        annotation = 1,
        One_step=args.
    )

    confusion_matrix_2, all_seqs2, _, _ = generate_rewind_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        cache_path="final_rewind_cache_oxe_pos_end_2.pkl",
        args = model_args,
        annotation = 2,
        One_step=args.
    )

    confusion_matrix_3, all_seqs3, _, _ = generate_rewind_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        cache_path="final_rewind_cache_oxe_pos_end_3.pkl",
        args = model_args,
        annotation = 3,
        One_step=args.
    )


    confusion_matrix_all_fail, _, _, _ = generate_rewind_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval_fail.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        cache_path="final_rewind_cache_oxe_pos_end_fail.pkl",
        args = model_args,
        One_step=args.
    )

    confusion_matrix_close_success, _, _, _ = generate_rewind_data(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval_close_succ.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        cache_path="final_rewind_cache_oxe_pos_end_close_succ.pkl",
        args = model_args,
        One_step=args.
    )

    generate_rewind_gif(
        h5_path="/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_dino_embeddings_eval.h5",
        json_path="new_task_v2.json",
        set_type="eval",
        rewind_model=rewind_model,
        device="cuda",
        args=model_args,
    )


    wandb.init(project="roboclip-v2", name=f"eval_rewind_new")
    
    # ============ 2) 计算相关(只用对角线) ============
    compute_pearson_correlation_from_sequences(
        all_seqs=all_seqs,
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


    # # ============ 4) 计算 MSE ============
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

    compute_spearman_correlation_from_sequences(
        all_seqs=all_seqs1,
        env_names=tasks,
        set_type="eval"
    )

    compute_spearman_correlation_from_sequences(
        all_seqs=all_seqs2,
        env_names=tasks,
        set_type="eval"
    )

    compute_spearman_correlation_from_sequences(
        all_seqs=all_seqs3,
        env_names=tasks,
        set_type="eval"
    )

    compute_spearman_correlation_multi_annotations(
        all_seqs_a=all_seqs1,
        all_seqs_b=all_seqs2,
        all_seqs_c=all_seqs3,
        all_seqs_d=all_seqs,
        env_names=tasks,
        set_type="eval"
    )

    rank_comparison(confusion_matrix_all_fail, confusion_matrix_close_success, confusion_matrix)
    wandb.finish()