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


os.environ["TOKENIZERS_PARALLELISM"] = "False"

def generate_liv_data(
    h5_path: str,
    json_path: str,
    set_type: str,
    rewind_model: torch.nn.Module,
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
                    frames_embed = torch.stack([
                        embedding_image(liv, processor, Image.fromarray(frame.astype(np.uint8))).squeeze()
                        for frame in frames_data
                    ], dim=0)
                    text_embed = embedding_text(liv, tokenizer, text_str)
                    print(frames_embed.shape, text_embed.shape)
                    video_reward = F.cosine_similarity(frames_embed, text_embed, dim=1)

                    cache_dict[cache_key_full] = video_reward
                    save_cache()
                    exit(0)
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
                        # 对 partial_count in [1..total_frames]
                        for partial_count in range(1, total_frames + 1):
                            cache_key_partial = (env_video_name, text_str, partial_count)
                            if cache_key_partial in cache_dict:
                                sub_reward = cache_dict[cache_key_partial]
                            else:
                                # 需要计算
                                # sub_frames => frames_data[:partial_count]
                                sub_frames = frames_data[:partial_count]
                                sub_frames_embed = torch.stack([
                                    embedding_image(liv, processor, Image.fromarray(frame.astype(np.uint8))).squeeze()
                                    for frame in sub_frames
                                ], dim=0)
                                text_embed = embedding_text(liv, tokenizer, text_str).squeeze()
                                sub_triangular_mask = torch.zeros((128, 128))
                                for j in range(partial_count):
                                    sub_triangular_mask[j, :j+1] = 1
                                sub_mask = torch.zeros(128)
                                sub_mask[:partial_count] = 1
                                sub_reward, _ = rewind_model(sub_frames_embed.unsqueeze(0), sub_triangular_mask.unsqueeze(1).unsqueeze(1), text_embed, sub_mask)
                                cache_dict[cache_key_partial] = sub_reward
                                save_cache()

                            predicted_seq.append(sub_reward)
                            
                    print(f"🎬 {env_video_name} ({text_str}): {predicted_seq}")
                    

                    predicted_sequences[i] = predicted_seq
                    print(f"🎬 : {predicted_sequences[i]}")

        return confusion_matrix, predicted_sequences, tasks, text_list


def main(eval_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    wandb.init(
        project=WANDB_PROJECT_NAME, 
        entity=WANDB_ENTITY_NAME, 
        group="OfflineEvalDecoder",
        name="confusion_matrix"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, tokenizer = load_model("liv")
    if eval_set == "train":
        eval_envs = json.load(open("task_subset.json"))["subset_6"]
        text = json.load(open("task_subset.json"))["train_annotation"]
        display_text = json.load(open("task_subset.json"))["train_dis_annotation"]
    elif eval_set == "eval":
        eval_envs = json.load(open("task_subset.json"))["evaluate_tasks"]
        text = json.load(open("task_subset.json"))["eval_annotation"]
        display_text = json.load(open("task_subset.json"))["eval_dis_annotation"]
    text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
    text_embeddings = normalize_embeddings(text_embeddings) # 10, 1024


    predicted_progress_row = []
    h5_file = h5py.File("metaworld_embedding_1_demo_dataset1.h5", 'r')


    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        

        traj_data = np.asarray(h5_file[env])

        traj_data = torch.tensor(traj_data).to(device).float()
        traj_data = traj_data[-1:]

        traj_data = normalize_embeddings(traj_data).repeat(text_embeddings.shape[0], 1)
        cos_sim = torch.nn.CosineSimilarity(dim=1)(traj_data, text_embeddings).detach().cpu().numpy()
        predicted_progress_row.append(cos_sim)


    predicted_progress_row = np.array(predicted_progress_row)
    eval_envs = [eval_envs[i].split("-v")[0] for i in range(len(eval_envs))]
    img = plot_matrix_as_image(predicted_progress_row, eval_envs, eval_set, display_text)


if __name__ == "__main__":
    main("train")
    main("eval")

    
    


