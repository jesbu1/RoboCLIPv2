import numpy as np
import torch
import torch.nn.functional as F
import h5py
import joblib
from clip_utils import normalize_embeddings, TwoLayerMLP, compute_M, SingleLayerMLP
import json
from clip_utils import load_model, embedding_text, embedding_image
from tqdm import tqdm


def mrr_clip_liv(gt_id, transform_model, text_pca_model, image_pca_model, linear_model, subtract=False, video_embeddings=None, text_embeddings=None):
    device = next(transform_model.parameters()).device
    N_v, D = video_embeddings.shape # N_v: number of videos, D: dimension of video embeddings
    N_t, _ = text_embeddings.shape # N_t: number of texts

    rewards = np.zeros((N_v, N_t))

    ground_truth_index = gt_id

    if text_embeddings is not None:
        text_embeddings = text_pca_model.transform(text_embeddings.detach().cpu().numpy())
        text_embeddings = torch.tensor(text_embeddings).to(device).float()


    # 遍历每一个视频嵌入 V[i]
    for i in range(N_v):
        video_input = video_embeddings[i:i+1]
        if image_pca_model is not None:
            video_input = image_pca_model.transform(video_input.detach().cpu().numpy())
            video_input = torch.tensor(video_input).to(device).float()
            video_input = linear_model(video_input)

        video_input = video_input.repeat(N_t, 1)

        if subtract:
            input_embedding = video_input - text_embeddings
        else:
            input_embedding = torch.cat([video_input, text_embeddings], dim=1)

        predicted_progress = transform_model(input_embedding).squeeze().detach().cpu().numpy()
        rewards[i] = predicted_progress


    def calculate_mrr(rewards, ground_truth_index, top_k=None):
        num_videos = rewards.shape[0]
        mrr_total = 0.0

        for i in range(num_videos):
            # 找出 reward 排名（从高到低排序）
            sorted_indices = np.argsort(rewards[i])[::-1]
            # 限制只考虑前 k 个排名 (top_k)
            if top_k is not None:
                sorted_indices = sorted_indices[:top_k]

            if ground_truth_index in sorted_indices:
                rank = np.where(sorted_indices == ground_truth_index)[0][0] + 1
                mrr_total += 1.0 / rank
        
        return mrr_total / num_videos

    mrr_top_1 = calculate_mrr(rewards, ground_truth_index, top_k=1)
    # print(f"MRR (Top 1): {mrr_top_1}")
    mrr_top_3 = calculate_mrr(rewards, ground_truth_index, top_k=3)
    # print(f"MRR (Top 3): {mrr_top_3}")
    mrr_top_5 = calculate_mrr(rewards, ground_truth_index, top_k=5)
    # print(f"MRR (Top 5): {mrr_top_5}")

    return mrr_top_1, mrr_top_3, mrr_top_5

if __name__ == "__main__":
    experiment_name = "RegressionLoss_liv_mse_pca_1.0_subtract"
    if "subtract" in experiment_name:
        subtract = True
    else:
        subtract = False
    h5_file_path = '/scr/jzhang96/metaworld_25_for_clip_liv.h5'
    h5_file = h5py.File(h5_file_path, "r")
    model_name = "liv"

    model_path = f"/home/jzhang96/RoboCLIPv2/clip/save_models/{experiment_name}/model_19.pth"

    if "pca" in experiment_name:
        
        text_pca_model_name = f"/home/jzhang96/RoboCLIPv2/clip/pca_models/{experiment_name}_text.pkl"
        image_pca_model_name = f"/home/jzhang96/RoboCLIPv2/clip/pca_models/{experiment_name}_image.pkl"

        text_pca_model = joblib.load(text_pca_model_name)
        image_pca_model = joblib.load(image_pca_model_name)
        if subtract:
            transform_model = TwoLayerMLP(image_pca_model.components_.shape[0])
        else:
            transform_model = TwoLayerMLP(image_pca_model.components_.shape[0] * 2)

        linear_model = SingleLayerMLP(image_pca_model.components_.shape[0], text_pca_model.components_.shape[0])
        linear_model.load_state_dict(torch.load(model_path)["linear_model"])
        linear_model.eval()
        linear_model = linear_model.to("cuda")

    else:
        if subtract:
            transform_model = TwoLayerMLP(1024)
        else:
            transform_model = TwoLayerMLP(1024 * 2)

    transform_model.load_state_dict(torch.load(model_path)["transform_model"])
    transform_model.eval()
    transform_model = transform_model.to("cuda")

    task_subset = json.load(open("task_subset.json"))
    train_subset = task_subset["subset_6"]
    eval_subset = task_subset["evaluate_tasks"]

    training_ann = task_subset["train_annotation"]
    eval_ann = task_subset["eval_annotation"]

    liv_model, liv_tokenizer, liv_preprocess = load_model("liv")

    train_ann_embeddings = embedding_text(liv_model, liv_tokenizer, training_ann)
    eval_ann_embeddings = embedding_text(liv_model, liv_tokenizer, eval_ann)
    training_ann_embeddings = normalize_embeddings(train_ann_embeddings)
    eval_ann_embeddings = normalize_embeddings(eval_ann_embeddings)


    video_group = h5_file[model_name]
    mrr_top_1_list = []
    mrr_top_3_list = []
    mrr_top_5_list = []
    for i in tqdm(range (len(train_subset))):
        task = train_subset[i]
        video_embeddings = []
        env_group = video_group[task]
        for key in sorted(list(env_group.keys())[:15], key=int):
            video_embeddings.append(np.array(env_group[key][:]))
        video_embeddings = np.concatenate(video_embeddings, axis=0)
        video_embeddings = normalize_embeddings(torch.from_numpy(video_embeddings).to("cuda"))

        mrr_top_1, mrr_top_3, mrr_top_5 = mrr_clip_liv(i, transform_model, text_pca_model, image_pca_model, 
                                                       linear_model, subtract=subtract, video_embeddings=video_embeddings, text_embeddings=training_ann_embeddings)
        mrr_top_1_list.append(mrr_top_1)
        mrr_top_3_list.append(mrr_top_3)
        mrr_top_5_list.append(mrr_top_5)
    print("Train MRR")
    print(f"Train Average MRR (Top 1): {np.mean(mrr_top_1_list)}")
    print(f"Train Average MRR (Top 3): {np.mean(mrr_top_3_list)}")
    print(f"Train Average MRR (Top 5): {np.mean(mrr_top_5_list)}")

    mrr_top_1_list = []
    mrr_top_3_list = []
    mrr_top_5_list = []
    for i in tqdm(range (len(eval_subset))):
        task = eval_subset[i]
        video_embeddings = []
        env_group = video_group[task]
        for key in sorted(list(env_group.keys())[:15], key=int):
            video_embeddings.append(np.array(env_group[key][:]))
        video_embeddings = np.concatenate(video_embeddings, axis=0)
        video_embeddings = normalize_embeddings(torch.from_numpy(video_embeddings).to("cuda"))

        mrr_top_1, mrr_top_3, mrr_top_5 = mrr_clip_liv(i, transform_model, text_pca_model, image_pca_model, 
                                                       linear_model, subtract=subtract, video_embeddings=video_embeddings, text_embeddings=eval_ann_embeddings)
        mrr_top_1_list.append(mrr_top_1)
        mrr_top_3_list.append(mrr_top_3)
        mrr_top_5_list.append(mrr_top_5)
    print("Eval MRR")
    print(f"Eval Average MRR (Top 1): {np.mean(mrr_top_1_list)}")
    print(f"Eval Average MRR (Top 3): {np.mean(mrr_top_3_list)}")
    print(f"Eval Average MRR (Top 5): {np.mean(mrr_top_5_list)}")

    h5_file.close()

    







    


    


        

    




    
    
    