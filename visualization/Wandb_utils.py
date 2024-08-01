import imageio
import torch as th
import csv
import numpy as np
from pca import plot_embeddings_3d, plot_embeddings, check_pairs, plot_distribution_histograms
from sklearn.metrics.pairwise import cosine_similarity
import wandb
import imageio
from Dataload import preprocess_human_demo, adjust_frames, readGif, Embedding_gpu
import json
import random
from collections import defaultdict
from meta_world_name_ann import task_ann
import os
import shutil
import joblib
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from torch.utils.data import Dataset
from mlp import normalize_embeddings
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:, :, None].to(device) #.cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)

def plot_s3d(video_embeddings, text_embeddings, choose_task, mappings, wandb_log, Train_or_Validate ,video_sample_per_task=15):
    if isinstance(video_embeddings, torch.Tensor):
        video_embeddings = video_embeddings.numpy()
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.numpy()

    pca = PCA(n_components=2)
    all_embeddings = np.concatenate((video_embeddings, text_embeddings), axis=0)
    reduced_embeddings = pca.fit_transform(all_embeddings)
    print("reduced_embeddings: ", reduced_embeddings.shape)

    index_to_video_id = mappings['index_to_video_id']
    index_to_text_label = mappings['index_to_text_label']

    # Plot the reduced embeddings
    colors = plt.cm.tab10(np.linspace(0, 1, len(choose_task)))  # 28 groups
    plt.figure(figsize=(10, 8))  # 创建一个新的图像

    for i in range(len(choose_task)):
        # Plot video embeddings
        video_group_data = reduced_embeddings[i * video_sample_per_task:(i + 1) * video_sample_per_task]
        x_video = video_group_data[:, 0]
        y_video = video_group_data[:, 1]
        video_ids = [index_to_video_id[i * video_sample_per_task + j] for j in range(video_sample_per_task)]
        label = video_ids[0].split('_output')[0]

        plt.scatter(x_video, y_video, color=colors[i], label=label)

        # Plot text embeddings
        text_index = len(video_embeddings) + i * video_sample_per_task
        if text_index < len(reduced_embeddings):
            text_x = reduced_embeddings[text_index, 0]
            text_y = reduced_embeddings[text_index, 1]
            plt.scatter(text_x, text_y, color=colors[i], marker='+')

    plt.title(f'Different Scenes {Train_or_Validate} PCA Visualization')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')

    plt.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 1, 1])  # adjust the plot to the right (to fit the legend)

    # 将图像上传到 WandB
    #wandb.log({"PCA Visualization": wandb.Image(plt)})
    wandb_log[f"{Train_or_Validate} PCA Visualization"] = wandb.Image(plt)



class MetaDataset(Dataset):
    def __init__(self, video_folder_path, indices, transform=None, num_samples=75, seed=42):
        self.video_folder_path = video_folder_path
        self.transform = transform
        self.num_samples = num_samples
        self.video_paths = []
        self.labels = []
        self.video_ids = []
        self.seed = seed

        with open('task_id.json', 'r') as f:
            folder_ann = json.load(f)

        random.seed(self.seed)

        for index in indices:
            task_name = folder_ann[str(index)]
            #print(task_name)
            task_folder = os.path.join(video_folder_path, str(index))
            #print(task_folder)
            if os.path.isdir(task_folder):
                video_files = [f for f in os.listdir(task_folder)]
                #print(len(video_files))
                if len(video_files) >= num_samples:
                    selected_videos = random.sample(video_files, self.num_samples)
                    #print(len(selected_videos))
                    self.video_paths.extend([os.path.join(task_folder, f) for f in selected_videos])
                    label = task_ann.get(task_name, "Unknown task")
                    #print(label)
                    self.labels.extend([label] * len(selected_videos))
                    simplified_task_name = task_name.split('-v2')[0]
                    self.video_ids.extend([f"{simplified_task_name}_{os.path.splitext(f)[0]}" for f in selected_videos])

        assert len(self.video_paths) == len(self.labels) == len(self.video_ids), "Mismatch between video paths, labels, and video IDs"

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self.readGIF(video_path)
        frames = preprocess_human_demo(frames)
        frames = adjust_frames(frames)

        if self.transform:
            frames = self.transform(frames)

        video_id = self.video_ids[idx]
        text_label = self.labels[idx]
        sample = {'video': frames, 'video_id': video_id, 'text': text_label}
        return sample

    def readGIF(self, file_path):
        reader = imageio.get_reader(file_path)
        frames = []
        for frame in reader:
            frames.append(np.array(frame))
        return np.stack(frames)


class Augmented_Batched_Dataset(Dataset):
    def __init__(self, base_dataset, transform, num_augmented_samples=99):
        self.base_dataset = base_dataset
        self.transform = transform
        self.num_augmented_samples = num_augmented_samples
        self.base_videos = torch.stack([sample['video'] for sample in base_dataset])
        # self.base_texts = [sample['text'] for sample in base_dataset]
        # self.base_video_ids = [sample['video_id'] for sample in base_dataset]

        self.augmented_videos_list = []
        for _ in range(num_augmented_samples):
            augmented_videos = self.transform(self.base_videos)
            self.augmented_videos_list.append(augmented_videos)

        # 将增强样本列表堆叠成一个张量
        self.augmented_videos = torch.cat(self.augmented_videos_list, dim=0)
        print('augmented_dataset', self.augmented_videos.shape)

    def __len__(self):
        return len(self.base_dataset) * (1 + self.num_augmented_samples)

    def __getitem__(self, idx):
        base_idx = idx // (1 + self.num_augmented_samples)
        aug_idx = idx % (1 + self.num_augmented_samples)

        sample = self.base_dataset[base_idx]

        augmented_sample = {
            'video': sample['video'],
            'text': sample['text'],
            'video_id': sample['video_id']
        }

        if aug_idx > 0 and self.transform:
            augmented_idx = base_idx + (aug_idx - 1) * len(self.base_videos)
            sample['video'] = self.augmented_videos[augmented_idx]
        print('augmented sample', augmented_sample['video'].shape)
        return augmented_sample


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, transform, device, num_augmented_samples=99):
        self.base_dataset = base_dataset
        self.transform = transform
        self.num_augmented_samples = num_augmented_samples
        self.cached_videos = []
        self.device = device
        for sample in base_dataset:
            video_on_gpu = sample['video'].to(self.device)
            self.cached_videos.append({
                'video': video_on_gpu,
                'text': sample['text'],
                'video_id': sample['video_id']
            })

    def __len__(self):
        return len(self.base_dataset) * (1 + self.num_augmented_samples)

    def __getitem__(self, idx):
        base_idx = idx // (1 + self.num_augmented_samples)# 商 0到9
        aug_idx = idx % (1 + self.num_augmented_samples)#余数 0到99

        sample = self.base_dataset[base_idx]
        #sample = self.cached_videos[base_idx]

        augmented_sample = {
            'video': sample['video'],
            'text': sample['text'],
            'video_id': sample['video_id']
        }

        if aug_idx > 0 and self.transform:
            augmented_sample['video'] = self.transform(augmented_sample['video'])

        return augmented_sample


def Test_Model(video_embeddings, text_embeddings, train_mappings, epoch, model, val_task, Train_or_Validate, wandb_log):
    if model != None:
        with th.no_grad():
            adjusted_video_embeddings = model(video_embeddings)
    else:
        adjusted_video_embeddings = video_embeddings
    
    # adjusted_video_embeddings = normalize_embeddings(adjusted_video_embeddings)
    # text_embeddings = normalize_embeddings(text_embeddings)
    
    accuracies_text, mrr_text = check_pairs(
        adjusted_video_embeddings.cpu().numpy(),
        text_embeddings.cpu().numpy(),
        train_mappings,
        False,
    )
    accuracies_video, mrr_video = check_pairs(
        adjusted_video_embeddings.cpu().numpy(),
        text_embeddings.cpu().numpy(),
        train_mappings,
        False,
        user='video'
    )
    #video_1, video_3, _ = compute_similarity_sorted(adjusted_video_embeddings, text_embeddings, val_task, 15)
    simi_score = th.mean(th.diag(th.matmul(adjusted_video_embeddings.cpu().float(), text_embeddings.cpu().float().t())))
    true_paired_cos_simi = F.cosine_similarity(adjusted_video_embeddings, text_embeddings, dim=1)

    # task_similarities = defaultdict(list)
    # for i, cos_sim in enumerate(true_paired_cos_simi):
    #     video_id = train_mappings["index_to_video_id"][i]
    #     task_id = video_id.split('_')[0]
    #     task_similarities[task_id].append(cos_sim.item())
    # average_task_similarities_for_chart = {task_id: np.mean(sims) for task_id, sims in task_similarities.items()}
    # average_task_similarities = [{"task_id": task_id, "average_cosine_similarity": np.mean(sims)}
    #                              for task_id, sims in task_similarities.items()]

    video_norm = th.mean(th.norm(adjusted_video_embeddings, dim=1))
    text_norm = th.mean(th.norm(text_embeddings, dim=1))
    milnce_loss = MILNCELoss()
    # validation_loss = milnce_loss(adjusted_video_embeddings, text_embeddings)
    validation_loss = milnce_loss(text_embeddings, adjusted_video_embeddings)

    wandb_log[f'{Train_or_Validate}_epoch'] = epoch + 1
    wandb_log[f'{Train_or_Validate}_loss'] = validation_loss
    wandb_log[f'{Train_or_Validate}_accuracy_top_1'] = accuracies_text.get("Top 1", "")
    wandb_log[f'{Train_or_Validate}_accuracy_top_3'] = accuracies_text.get("Top 3", "")
    wandb_log[f'{Train_or_Validate}_accuracy_top_5'] = accuracies_text.get("Top 5", "")
    wandb_log[f'{Train_or_Validate}_accuracy_top_10'] = accuracies_text.get("Top 10", "")
    wandb_log[f'{Train_or_Validate}_mrr_top_1'] = mrr_text.get("Top 1", "")
    wandb_log[f'{Train_or_Validate}_mrr_top_3'] = mrr_text.get("Top 3", "")
    wandb_log[f'{Train_or_Validate}_mrr_top_5'] = mrr_text.get("Top 5", "")
    wandb_log[f'{Train_or_Validate}_mrr_top_10'] = mrr_text.get("Top 10", "")
    wandb_log[f'{Train_or_Validate}_video_accuracy_top_1'] = accuracies_video.get("Top 15", "")
    wandb_log[f'{Train_or_Validate}_video_accuracy_top_3'] = accuracies_video.get("Top 45", "")
    wandb_log[f'{Train_or_Validate}_video_accuracy_top_5'] = accuracies_video.get("Top 75", "")
    wandb_log[f'{Train_or_Validate}_video_mrr_top_1'] = mrr_video.get("Top 15", "")
    wandb_log[f'{Train_or_Validate}_video_mrr_top_3'] = mrr_video.get("Top 45", "")
    wandb_log[f'{Train_or_Validate}_video_mrr_top_5'] = mrr_video.get("Top 75", "")
    wandb_log[f'{Train_or_Validate}_video_norm'] = video_norm
    wandb_log[f'{Train_or_Validate}_text_norm'] = text_norm
    wandb_log[f'{Train_or_Validate}_similarity_score'] = simi_score.item()
    wandb_log[f'{Train_or_Validate}_cosine_similarity'] = true_paired_cos_simi.mean().item()
    #wandb_log[f'{Train_or_Validate}_average_task_similarities_chart'] = average_task_similarities_for_chart

    # if epoch == 999:
    #     wandb_table = wandb.Table(columns=["task_id", "average_cosine_similarity"])
    #     for entry in average_task_similarities:
    #         wandb_table.add_data(entry["task_id"], entry["average_cosine_similarity"])
    #     wandb_log[f'{Train_or_Validate}_table_average_task_similarities'] = wandb_table

    #if Train_or_Validate == 'Validate':
    plot_s3d(adjusted_video_embeddings.cpu(), text_embeddings.cpu(),
             val_task, train_mappings, wandb_log, Train_or_Validate, 15)
        #wandb_log[f'{Train_or_Validate}_average_task_similarities_chart'] = average_task_similarities_for_chart
    return adjusted_video_embeddings, text_embeddings


def generate_augmented_embeddings(video_embeddings, text_embeddings, num_augmented, augment_fn):
    original_count, embedding_dim = video_embeddings.shape
    augmented_video_embeddings = []
    augmented_text_embeddings = []

    for i in range(original_count):
        original_video_embedding = video_embeddings[i].unsqueeze(0)
        original_text_embedding = text_embeddings[i]

        augmented_video_embeddings.append(original_video_embedding.squeeze(0))
        augmented_text_embeddings.append(original_text_embedding)

        for _ in range(num_augmented):
            augmented_video_embedding = augment_fn(original_video_embedding).squeeze(0)
            augmented_video_embeddings.append(augmented_video_embedding)
            augmented_text_embeddings.append(original_text_embedding)

    return th.stack(augmented_video_embeddings), th.stack(augmented_text_embeddings)


import kornia.augmentation as K
import torch.nn as nn
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
aug_cfg = AttrDict(
    brightness=0.3,
    contrast=0.3,
    saturation=0.3,
    hue=0.3,
    color_p=0.9,
    noise_std=0.1,
    noise_p=0.0,
    channel_shuffle_p=0.0,
    degrees=15,
    translate=0.1,
    affine_p=0.6,
    erase_p=0.1,
)

image_aug_fn = nn.Sequential(
    K.ColorJitter(
        aug_cfg.brightness,
        aug_cfg.contrast,
        aug_cfg.saturation,
        aug_cfg.hue,
        p=aug_cfg.color_p,
    ),
    K.RandomGaussianNoise(std=aug_cfg.noise_std, p=aug_cfg.noise_p),
    K.RandomChannelShuffle(p=aug_cfg.channel_shuffle_p),
    K.RandomAffine(
        degrees=aug_cfg.degrees,
        translate=(aug_cfg.translate, aug_cfg.translate),
        p=aug_cfg.affine_p,
    ),
    K.RandomErasing(p=aug_cfg.erase_p),
)


def generate_augmented_dataset(dataset, num_augmented, augment_fn):
    augmented_samples = []

    for item in dataset:
        original_video = item['video']
        original_text = item['text']
        video_id = item['video_id']

        # 原始视频形状为 (1, 3, 32, 224, 224)
        batch_size, channels, num_frames, height, width = original_video.shape

        # 添加原始视频和文本
        augmented_samples.append({'video': original_video, 'text': original_text, 'video_id': video_id})

        for _ in range(num_augmented):
            # 将视频张量从 (1, 3, 32, 224, 224) 转换为 (32, 3, 224, 224)
            original_video_permuted = original_video.squeeze(0).permute(1, 0, 2, 3)  # (32, 3, 224, 224)
            # 应用增强函数
            augmented_video = augment_fn(original_video_permuted)  # (32, 3, 224, 224)
            # 转换回 (1, 3, 32, 224, 224)
            augmented_video = augmented_video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 32, 224, 224)
            print('augmented_video', augmented_video.shape)
            augmented_samples.append({'video': augmented_video, 'text': original_text, 'video_id': video_id})

    augmented_dataset = AugmentedDataset(augmented_samples)

    return augmented_dataset


def reduce_dimension(
        embeddings, variance_threshold, train_size, embed_type, seed, strong, num, dimension=None, filter=False, pca_emb=None, kernel = 'linear', val_task_name=None
):
    if variance_threshold == 0:
        return None, embeddings.float()
    if kernel =='linear':
        if dimension:
            pca = PCA(n_components=dimension)
        else:
            pca = PCA(n_components=variance_threshold)
    else:
        if dimension:
            pca = KernelPCA(n_components=dimension, kernel=kernel)
        else:
            pca = KernelPCA(n_components=variance_threshold, kernel=kernel)
    if pca_emb != None:
        pca.fit(pca_emb)
        reduced_embeddings = pca.transform(embeddings)
    else:
        reduced_embeddings = pca.fit_transform(embeddings)
    # os.makedirs('saved_model/M/metaworld/pca_model', exist_ok=True)
    pca_save_path = (f"/scr/yusenluo/RoboCLIP/visualization/saved_model/pca_loss_models/{val_task_name}_Seed_{seed}/"
                     f"{variance_threshold}_Aug_{strong}_{train_size}_{kernel}")
    os.makedirs(pca_save_path, exist_ok=True)
    if filter:
        model_filename = (
            f"saved_model/M/OpenX/droid/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"{pca_save_path}/pca_model_{embed_type}.pkl"
        )
    joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return pca, torch.from_numpy(reduced_embeddings).float()


def normalize_and_pca(sampled_video_embeddings, sampled_text_embeddings, validate_video_embeddings_normalized, validate_text_embeddings_normalized, variance_threshold, 
                      current_sample_size, seed, device, strong, pca_sample_size, kernel = 'linear', val_task_name=None):
    train_video_embeddings_normalized = normalize_embeddings(
        sampled_video_embeddings
    ).clone().cpu()
    train_text_embeddings_normalized = normalize_embeddings(
        sampled_text_embeddings
    ).clone().cpu()
    pca_train_alltext = th.cat((train_text_embeddings_normalized, validate_text_embeddings_normalized.cpu()), dim=0)
    print(pca_train_alltext.shape)
    pca_text, reduced_train_text = reduce_dimension(train_text_embeddings_normalized, variance_threshold,
                                                    current_sample_size,
                                                    'text', seed=seed, strong=strong, num=pca_sample_size,
                                                    filter=False, kernel=kernel, val_task_name=val_task_name) #pca_emb=pca_train_alltext
    pca_video, reduced_train_video = reduce_dimension(train_video_embeddings_normalized, variance_threshold,
                                                      current_sample_size, 'video', filter=False,
                                                      dimension=reduced_train_text.shape[1],
                                                      seed=seed, strong=strong, num=pca_sample_size,kernel=kernel, val_task_name=val_task_name)  # 35，512
    if pca_text != None:
        reduced_validate_video = torch.from_numpy(
            pca_video.transform(validate_video_embeddings_normalized.cpu())).float().to(device)
        reduced_validate_text = torch.from_numpy(
            pca_text.transform(validate_text_embeddings_normalized.cpu())).float().to(device)
    else:
        reduced_validate_video = (validate_video_embeddings_normalized).float().to(device)
        reduced_validate_text = (validate_text_embeddings_normalized).float().to(device)
    reduced_train_text = normalize_embeddings(reduced_train_text).to(device)
    reduced_train_video = reduced_train_video.to(device)
    print(reduced_train_text.shape, reduced_train_video.shape)
    return reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text


def find_video(video_id, target_folder):
    # for sampled_text in sampled_texts:
    #     # 根据给定的文本标签生成相应的视频文件路径
    #     video_id = sampled_text.replace(' ', '_')  # 替换空格为下划线
    video_path = os.path.join('/scr/yusenluo/RoboCLIP/OpenX/droid', f"{video_id}.gif")

    # 检查视频文件是否存在
    if os.path.exists(video_path):
        # 复制文件到目标文件夹
        shutil.copy(video_path, target_folder)
        print(f"Copied {video_path} to {target_folder}")
    else:
        print(f"Video file {video_path} does not exist")


def save_augmented_videos(dataset, save_path, num_videos=10):
    os.makedirs(save_path, exist_ok=True)
    sampled_indices = torch.randperm(len(dataset))[:num_videos]
    for i in sampled_indices:
        sample = dataset[i]
        video = sample['video'].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (32, 224, 224, 3)

        video = (video * 255).astype(np.uint8)

        video_id = sample['video_id']
        save_file = os.path.join(save_path, f"{video_id}_augmented_{i}.gif")
        find_video(video_id, save_path)
        imageio.mimsave(save_file, video, fps=5)
        print(f"Saved {save_file}")


def reduce_with_pca(augmented_video_embeddings, augmented_text_embeddings, validate_video_embeddings_normalized,
                                      validate_text_embeddings_normalized, pca_video, pca_text, device):
    train_video_embeddings_normalized = normalize_embeddings(
        augmented_video_embeddings
    ).clone().cpu()
    train_text_embeddings_normalized = normalize_embeddings(
        augmented_text_embeddings
    ).clone().cpu()
    reduced_validate_video = torch.from_numpy(
        pca_video.transform(validate_video_embeddings_normalized.cpu())).float().to(device)
    reduced_validate_text = torch.from_numpy(
        pca_text.transform(validate_text_embeddings_normalized.cpu())).float().to(device)
    reduced_train_video = torch.from_numpy(
        pca_video.transform(train_video_embeddings_normalized)).float().to(device)
    reduced_train_text = torch.from_numpy(
        pca_text.transform(train_text_embeddings_normalized)).float().to(device)
    return reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text
    
    


def Test_Model_OpenX(adjusted_video_embeddings, text_embeddings, train_mappings, model, plot_dir_name, sample_size_for_training):
    
    accuracies_original, mrr_original = check_pairs(
        adjusted_video_embeddings.cpu().numpy(),
        text_embeddings.cpu().numpy(),
        train_mappings,
        False,
    )
    csv_file_path = 'SubspaceAlignment_Result_OpenX.csv'
    simi_score = th.mean(th.diag(th.matmul(adjusted_video_embeddings.cpu().float(), text_embeddings.cpu().float().t())))
    cos_simi = np.mean(np.diag
                       (cosine_similarity(adjusted_video_embeddings.cpu().numpy(),
                                          text_embeddings.cpu().numpy())))
    video_norm = th.mean(th.norm(adjusted_video_embeddings, dim=1))
    text_norm = th.mean(th.norm(text_embeddings, dim=1))
    milnce_loss = MILNCELoss()
    validation_loss = milnce_loss(adjusted_video_embeddings, text_embeddings)

    data_to_write_original = [
        plot_dir_name, filter, len(adjusted_video_embeddings), 'No PCA',
        sample_size_for_training, 'Original', adjusted_video_embeddings.shape[1],
        accuracies_original.get("Top 1", ""), accuracies_original.get("Top 3", ""),
        accuracies_original.get("Top 5", ""),
        accuracies_original.get("Top 10", ""), mrr_original.get("Top 1", ""), mrr_original.get("Top 3", ""),
        mrr_original.get("Top 5", ""),
        mrr_original.get("Top 10", ""), simi_score.item(), cos_simi
    ]
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write_original)


def compute_similarity_sorted(video_embeddings, text_embeddings, task_names, video_sample_per_task):
    '''
    compute the similarity between video embeddings and text embeddings
    '''
    text_single_embeddings = text_embeddings[::video_sample_per_task]
    similarity = np.dot(video_embeddings, text_single_embeddings.T)

    # sort the similarity matrix, from high to low
    similarity_sorted = np.argsort(similarity, axis=1)[:, ::-1][:, :5]

    top1_accuracy = 0
    top3_accuracy = 0
    top5_accuracy = 0
    GT_label = np.zeros(len(video_embeddings))
    for i in range(len(task_names)):
        GT_label[i * video_sample_per_task:(i + 1) * video_sample_per_task] = i

    for i in range(len(GT_label)):
        label = GT_label[i]
        if label in similarity_sorted[i][:1]:
            top1_accuracy += 1
        if label in similarity_sorted[i][:3]:
            top3_accuracy += 1
        if label in similarity_sorted[i][:5]:
            top5_accuracy += 1

    top1_accuracy = top1_accuracy / len(GT_label)
    top3_accuracy = top3_accuracy / len(GT_label)
    top5_accuracy = top5_accuracy / len(GT_label)
    print(top1_accuracy, top3_accuracy, top5_accuracy)
    return top1_accuracy, top3_accuracy, top5_accuracy