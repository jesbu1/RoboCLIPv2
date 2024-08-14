import cv2
import imageio
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch as th
import h5py
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from s3dg import S3D
import torch.nn as nn
from sklearn.decomposition import PCA, KernelPCA
import torch.nn.functional as F
import random
from meta_world_name_ann import task_ann
device = th.device("cuda" if th.cuda.is_available() else "cpu")


# 预处理 S3D 输入 video的三个函数
def preprocess_human_demo(frames):
    """
    Preprocesses frames for video by adjusting size, adding a batch dimension,
    and transposing dimensions to match S3D model input requirements.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Preprocessed frames as a numpy array.
    """
    frames = np.array(frames)
    frames = adjust_size(frames)
    frames = frames[None, :,:,:,:]
    frames = frames.transpose(0, 4, 1, 2, 3)
    if frames.shape[1] > 3:
        frames = frames[:, :3]
    frames = frames / 255
    return frames

def adjust_frames(frames, target_frame_count = 32):
    """
    Ensures same numbers of frames(32).
    """
    _, _, frame_count, _, _ = frames.shape
    #print(f"frames number{frame_count}")
    frames = th.from_numpy(frames)
    if frame_count < target_frame_count:
        blank_frames = th.zeros(
            (frames.shape[0], frames.shape[1], target_frame_count - frame_count, frames.shape[3], frames.shape[4]),
            dtype=frames.dtype)
        adjusted_frames = th.cat((frames, blank_frames), dim=2)

    elif frame_count > target_frame_count:
        indices = th.linspace(0, frame_count - 1, target_frame_count, dtype=th.long)
        adjusted_frames = th.index_select(frames, 2, indices)
    else:
        adjusted_frames = frames

    return adjusted_frames

def adjust_size(frames):
    """
    Adjusts the size of the frames to a target height and width by cropping.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Cropped frames as a numpy array.
    """
    if len(frames) == 0:
        return np.array([])

    target_height = 224
    target_width = 224

    height, width, _ = frames[0].shape

    if height < target_height or width < target_width:
        adjusted_frames = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR) for frame in
                           frames]
    else:
        start_x = width // 2 - target_width // 2
        start_y = height // 2 - target_height // 2
        adjusted_frames = [frame[start_y:start_y + target_height, start_x:start_x + target_width] for frame in frames]

    return np.array(adjusted_frames)



#用于accuracy 和 mrr compute的函数，只调用check_pairs 就可以一起算
def check_pairs(
        reduced_video_embeddings: np.array,
        reduced_text_embeddings: np.array,
        mappings,
        small_scale=False,
        user='text'
):
    if user == 'text':
        similarities = reduced_text_embeddings @ reduced_video_embeddings.T
        k_list = [1, 3, 5, 10]
    else:
        similarities = reduced_video_embeddings @ reduced_text_embeddings.T
        k_list = [15, 45, 75]
    sorted_video_indices = np.argsort(-similarities, axis=1)

    video_id_to_text_label = mappings["video_id_to_text_label"]
    index_to_video_id = mappings["index_to_video_id"]
    index_to_text_label = mappings["index_to_text_label"]
    accuracies = {}
    mrr = {}

    ground_truth_labels = [video_id_to_text_label[index_to_video_id[i]] for i in range(len(reduced_video_embeddings))]

    mrr_k = mrr_score(similarities, ground_truth_labels, index_to_text_label, k_list=k_list)
    print("MRR Scores:", mrr_k)

    if small_scale:
        for n in [1, 3, 5]:
            correct_pairs = np.array(
                [
                    video_id_to_text_label[index_to_video_id[ground_truth]] in [index_to_text_label[idx] for idx in sorted_video_indices[i, :n]]
                    for i, ground_truth in enumerate(range(len(reduced_video_embeddings)))
                ]
            )
            accuracy = np.mean(correct_pairs)
            accuracies[f"Top {n}"] = round(accuracy * 100, 4)
            print(f"Top {n} accuracy: {accuracy * 100:.2f}%")
        top_1_indices = sorted_video_indices[:, 0]
        correct_top_1_pairs = [video_id_to_text_label[index_to_video_id[i]] == index_to_text_label[top_1_idx] for i, top_1_idx in enumerate(top_1_indices)]
        incorrect_top_1_indices = np.where(~np.array(correct_top_1_pairs))[0]

        incorrect_pair_text_id = [
            index_to_text_label[i] for i in incorrect_top_1_indices
        ]
        print(
            f"IDs of incorrectly paired text embeddings (Top 1): {incorrect_pair_text_id}"
        )

        for i, indices in enumerate(sorted_video_indices):
            text_id = index_to_text_label[i]
            original_video_id = video_id_to_text_label[text_id]
            sorted_video_labels = [index_to_video_id[j] for j in indices]
            print(f"Text {text_id}:")
            print(f"Ground truth video id: {original_video_id}")
            print(f"Sorted Matching video ids: {sorted_video_labels}")
    else:
        correct_pairs_info = []

        for n in k_list: #[1, 3, 5, 10]
            correct_pairs = []
            for i, indices in enumerate(sorted_video_indices):
                sorted_video_labels = [index_to_text_label[j] for j in indices[:n]]
                if index_to_text_label[i] in sorted_video_labels:
                    correct_pairs.append(True)
                    correct_pairs_info.append((index_to_text_label[i], ground_truth_labels[i]))
                else:
                    correct_pairs.append(False)
            accuracy = np.mean(correct_pairs)
            accuracies[f"Top {n}"] = round(accuracy * 100, 4)
            print(f"Accuracy within top {n}: {accuracy * 100:.2f}%")

    return accuracies, mrr_k
def mrr_score(similarities, ground_truth_labels, index_to_text_label, k_list=[1, 3, 5, 10]):
    """
    Calculate the Mean Reciprocal Rank (MRR) at multiple values of k for a set of embeddings.

    Parameters:
    - similarities (np.array): A 2D numpy array where each row represents the similarity scores
      between a video embedding and all text embeddings.
    - ground_truth_labels (list): A list where each element is the ground truth text label
      for the corresponding video embedding.
    - index_to_text_label (dict): A dictionary mapping from indices to text labels.
    - k_list (list): A list of integers specifying the k values to calculate MRR for.

    Returns:
    - dict: A dictionary where keys are k values and values are the MRR at each k.
    """
    sorted_indices = np.argsort(-similarities, axis=1)
    mrr_scores = {}

    for k in k_list:
        reciprocal_ranks = []

        for i, sorted_index_list in enumerate(sorted_indices):
            sorted_text_labels = [index_to_text_label[idx] for idx in sorted_index_list[:k]]
            if ground_truth_labels[i] in sorted_text_labels:
                rank = sorted_text_labels.index(ground_truth_labels[i]) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        mrr_scores[f"Top {k}"] = np.mean(reciprocal_ranks)

    return mrr_scores


class SingleLayerMLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


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


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


def get_s3d_embeddings(train_task_id, val_task_id, s3d, seed):
    """
    之前用的从gif得到s3d embeddings 和mapping 的方法，目前感觉比较麻烦
    """
    train_dataset = MetaDataset(
        "/scr/yusenluo/RoboCLIP/metaworld_generate_gifs", train_task_id, num_samples=15, seed=seed
    )
    val_dataset = MetaDataset(
        "/scr/yusenluo/RoboCLIP/metaworld_generate_gifs", val_task_id, num_samples=15, seed=seed
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=2
    )
    validate_data_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=2
    )
    # print(len(train_dataset))
    # print(len(val_dataset))
    train_video_embeddings, train_text_embeddings, _, train_mappings = Embedding_gpu(
        s3d, train_data_loader
    )
    validate_video_embeddings, validate_text_embeddings, _, validate_mappings = Embedding_gpu(
        s3d, validate_data_loader
    )
    validate_video_embeddings_normalized = normalize_embeddings(
        validate_video_embeddings
    ).clone()
    validate_text_embeddings_normalized = normalize_embeddings(
        validate_text_embeddings
    ).clone()
    train_video_embeddings_normalized = normalize_embeddings(
        train_video_embeddings
    ).clone()
    train_text_embeddings_normalized = normalize_embeddings(
        train_text_embeddings
    ).clone()

    return (train_video_embeddings_normalized, train_text_embeddings_normalized, validate_video_embeddings_normalized,
            validate_text_embeddings_normalized, train_mappings, validate_mappings)


def get_xclip_embeddings(task_id):
    """
    之前用的从你给我的h5文件得到xclip embeddings 和mapping 的方法
    """
    video_features = []
    train_task_name = []
    text_features = []
    video_ids = []
    text_labels = []
    h5_file_path = "/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5"
    h5_file = h5py.File(h5_file_path, 'r')
    for task in task_id:
        single_task = str(task)
        data_group = h5_file[single_task]
        task_name = data_group.attrs["task_name"].split("-v2")[0]
        print("train_task_name:", task_name)
        video_idx = len(list(data_group.keys()))
        text_label = data_group.attrs["task_annotation"]
        attrs = data_group.attrs
        # for key, value in attrs.items():
        #     print(f"{key}: {value}")
        choose_idx_range = random.sample(range(video_idx - 1),
                                         15)  # xclip_text_feature also is a key name
        this_video_feature = []
        this_text_feature = []
        for idx in choose_idx_range:
            video_feature = data_group[str(idx)]["xclip_video_feature"]
            #print(data_group[str(idx)].keys())
            video_id = f"{task_name}_{idx}"
            video_ids.append(video_id)
            text_labels.append(text_label)
            this_video_feature.append(video_feature)
            text_feature = data_group["xclip_text_feature"]
            this_text_feature.append(text_feature)

        this_video_feature = np.array(this_video_feature)
        this_text_feature = np.array(this_text_feature)
        video_features.append(this_video_feature)
        train_task_name.append(task_name)
        text_features.append(this_text_feature)
    mappings = {
        "video_id_to_text_label": dict(zip(video_ids, text_labels)),
        "index_to_video_id": {i: vid for i, vid in enumerate(video_ids)},
        "index_to_text_label": {i: lbl for i, lbl in enumerate(text_labels)}
    }
    # for mapping_name, mapping_dict in train_mappings.items():
    #     print(f"{mapping_name}:")
    #     for key, value in mapping_dict.items():
    #         print(f"  {key}: {value}")
    video_features = np.concatenate(video_features, axis=0)
    text_features = np.concatenate(text_features, axis=0)
    video_features = normalize_embeddings(th.from_numpy(video_features))
    text_features = normalize_embeddings(th.from_numpy(text_features))

    return video_features, text_features, mappings


def get_s3d_embeddings_h5(val_task_id):
    evaluate_h5 = h5py.File("/home/jzhang96/RoboCLIPv2/losses/metaworld_s3d_embedding.h5", "r")
    text_h5 = h5py.File("/scr/jzhang96/metaworld_s3d_text.h5", "r")
    with open('id_task.json', 'r') as f:
        task_mapping = json.load(f)
    matching_tasks = [task for task, id_str in task_mapping.items() if int(id_str) in val_task_id]

    video_ids = []
    text_labels = []
    evaluate_video_embeddings = []
    evaluate_text_embeddings = []
    for keys in matching_tasks:
        task_data = np.asarray(evaluate_h5[keys])
        # random choose 10
        choose_index = np.random.choice(task_data.shape[0], 15, replace=False)

        video_id = [f"{keys}_{index}" for index in choose_index]
        #print(video_id)
        video_ids.extend(video_id)
        text_label = [task_ann.get(keys, "Unknown task") for index in choose_index]
        #print(text_label)
        text_labels.extend(text_label)

        task_data = task_data[choose_index]
        evaluate_video_embeddings.append(task_data)

        text_embedding = [np.asarray(text_h5[keys]["embedding"]) for index in choose_index]
        evaluate_text_embeddings.append(text_embedding)
    #print(video_ids)
    evaluate_video_embeddings = np.concatenate(evaluate_video_embeddings, axis=0)
    evaluate_video_embeddings = normalize_embeddings(evaluate_video_embeddings)
    evaluate_video_embeddings = th.tensor(evaluate_video_embeddings).cuda()
    eval_text_embedding = np.concatenate(evaluate_text_embeddings, axis=0)
    eval_text_embedding = normalize_embeddings(eval_text_embedding)
    mappings = {
        "video_id_to_text_label": dict(zip(video_ids, text_labels)),
        "index_to_video_id": {i: vid for i, vid in enumerate(video_ids)},
        "index_to_text_label": {i: lbl for i, lbl in enumerate(text_labels)}
    }
    evaluate_h5.close()
    text_h5.close()
    return evaluate_video_embeddings, eval_text_embedding, mappings


#以下是pca 加 subspace alignment baseline 的三个函数
def reduce_dimension(
        embeddings, variance_threshold, embed_type, seed, dimension=None, pca_emb=None, kernel='linear', val_task_name=None
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
                     f"{variance_threshold}_{kernel}")
    os.makedirs(pca_save_path, exist_ok=True)
    model_filename = (
        f"{pca_save_path}/pca_model_{embed_type}.pkl"
    )

    joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return pca, th.from_numpy(reduced_embeddings).float()


def normalize_and_pca(sampled_video_embeddings, sampled_text_embeddings, validate_video_embeddings_normalized, validate_text_embeddings_normalized, variance_threshold,
                      seed, device, kernel='linear', val_task_name=None):
    train_video_embeddings_normalized = normalize_embeddings(
        sampled_video_embeddings
    ).clone().cpu()
    train_text_embeddings_normalized = normalize_embeddings(
        sampled_text_embeddings
    ).clone().cpu()
    pca_train_alltext = th.cat((train_text_embeddings_normalized, validate_text_embeddings_normalized.cpu()), dim=0)
    print(pca_train_alltext.shape)
    pca_text, reduced_train_text = reduce_dimension(train_text_embeddings_normalized, variance_threshold,
                                                    'text', seed=seed,
                                                    filter=False, kernel=kernel, val_task_name=val_task_name) #pca_emb=pca_train_alltext
    pca_video, reduced_train_video = reduce_dimension(train_video_embeddings_normalized, variance_threshold, 'video', filter=False,
                                                      dimension=reduced_train_text.shape[1], seed=seed,
                                                      kernel=kernel, val_task_name=val_task_name)  # 35，512
    if pca_text != None:
        reduced_validate_video = th.from_numpy(
            pca_video.transform(validate_video_embeddings_normalized.cpu())).float().to(device)
        reduced_validate_text = th.from_numpy(
            pca_text.transform(validate_text_embeddings_normalized.cpu())).float().to(device)
    else:
        reduced_validate_video = (validate_video_embeddings_normalized).float().to(device)
        reduced_validate_text = (validate_text_embeddings_normalized).float().to(device)
    reduced_train_text = normalize_embeddings(reduced_train_text).to(device)
    reduced_train_video = reduced_train_video.to(device)
    print(reduced_train_text.shape, reduced_train_video.shape)
    return reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text


def compute_M(X_S, X_T, variance_threshold, seed, filter=False):
    M = np.dot(X_S, X_T.T)  # 35 35
    M_tensor = th.from_numpy(M).float()
    # if filter:
    #     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/filter"
    # else:
    #     save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/pca_matrix_models"
    # os.makedirs(save_dir, exist_ok=True)
    # M_model_path = f"{save_dir}/M_model_{variance_threshold}_Seed{seed}.pth"
    # th.save(M_tensor, M_model_path)
    # print(f'M model saved to {M_model_path}')
    return M_tensor


class TextEmbedding:
    def __init__(self,
            token_to_word_path="./s3d_dict.npy",
            max_words=16,
            ):
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent.lower())) for sent in x]
        return th.stack(split_x, dim=0)


#使用GPU从s3d 中得到 embeddings
def Embedding_gpu(model, data_loader):
    """
    Generates embeddings for video and text using a given model and accumulates them along with video IDs and text labels into a dataset.

    Parameters:
    - model: The model used for generating embeddings.
    - data_loader: DataLoader providing batches of videos and corresponding text labels.

    Returns:
    - embeddings_dataset: A custom dataset containing all video embeddings, text embeddings, video IDs, and text labels.
    - mappings: A dictionary containing mappings from video IDs to text labels, and indices to video IDs and text labels.
    """
    all_video_embeddings = []
    all_text_embeddings = []
    video_ids, text_labels = [], []
    text_embedding = TextEmbedding(token_to_word_path="../s3d_dict.npy")
    for batch in data_loader:
        videos = th.squeeze(batch['video'].float(), dim=1)
        print(videos.shape)
        text_labels_batch = batch['text']
        video_ids_batch = batch['video_id']
        text_emb = text_embedding._words_to_ids(text_labels_batch)

        videos = videos.to(device)
        text_emb = text_emb.to(device)

        with th.no_grad():
            video_output = model(videos)
            text_output = model.text_module(text_emb)

            all_video_embeddings.append(video_output['video_embedding'])
            all_text_embeddings.append(text_output['text_embedding'])

        video_ids.extend(video_ids_batch)
        text_labels.extend(text_labels_batch)

    all_video_embeddings = th.cat(all_video_embeddings, dim=0)
    all_text_embeddings = th.cat(all_text_embeddings, dim=0)

    # Construct mappings
    mappings = {
        "video_id_to_text_label": dict(zip(video_ids, text_labels)),
        "index_to_video_id": {i: vid for i, vid in enumerate(video_ids)},
        "index_to_text_label": {i: lbl for i, lbl in enumerate(text_labels)}
    }

    # embeddings_dataset = EmbeddingsDataset(all_video_embeddings.cpu(), all_text_embeddings.cpu(), video_ids,
    #                                        text_labels)
    return all_video_embeddings, all_text_embeddings, None, mappings


#用于计算accuracy 和 mrr, norm, similarity 等一系列metric并上传wandb的
def eval_model(video_embeddings, text_embeddings, mappings, epoch, model, val_task, Train_or_Validate, wandb_log):
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
        mappings,
        False,
    )
    accuracies_video, mrr_video = check_pairs(
        adjusted_video_embeddings.cpu().numpy(),
        text_embeddings.cpu().numpy(),
        mappings,
        False,
        user='video'
    )
    # video_1, video_3, _ = compute_similarity_sorted(adjusted_video_embeddings, text_embeddings, val_task, 15)
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
    # wandb_log[f'{Train_or_Validate}_average_task_similarities_chart'] = average_task_similarities_for_chart

    # if epoch == 999:
    #     wandb_table = wandb.Table(columns=["task_id", "average_cosine_similarity"])
    #     for entry in average_task_similarities:
    #         wandb_table.add_data(entry["task_id"], entry["average_cosine_similarity"])
    #     wandb_log[f'{Train_or_Validate}_table_average_task_similarities'] = wandb_table

    # if Train_or_Validate == 'Validate':
    #我用来作可视图的函数，如已有，则可以替换
    # plot_s3d(adjusted_video_embeddings.cpu(), text_embeddings.cpu(),
    #          val_task, train_mappings, wandb_log, Train_or_Validate, 15)


    # wandb_log[f'{Train_or_Validate}_average_task_similarities_chart'] = average_task_similarities_for_chart
    return adjusted_video_embeddings, text_embeddings


#简化版evaluate mrr的函数
def eval_mrr(model, evaluate_task, video_embeddings, text_embeddings, mappings):
    if model != None:
        with th.no_grad():
            adjusted_video_embeddings = model(video_embeddings)
    else:
        adjusted_video_embeddings = video_embeddings

    # adjusted_video_embeddings = normalize_embeddings(adjusted_video_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)

    # text当user
    accuracies_text, mrr_text = check_pairs(
        adjusted_video_embeddings.cpu().numpy(),
        text_embeddings.cpu().numpy(),
        mappings,
        False,
    )

    # video 当user
    # accuracies_video, mrr_video = check_pairs(
    #     adjusted_video_embeddings.cpu().numpy(),
    #     text_embeddings.cpu().numpy(),
    #     mappings,
    #     False,
    #     user='video'
    # )

    #这里是返回accuracy，实际可以都要
    # return (accuracies_text.get("Top 1", ""), accuracies_text.get("Top 3", ""), accuracies_text.get("Top 5", ""),
    #         accuracies_text.get("Top 10", ""))
    return (mrr_text.get("Top 1", ""), mrr_text.get("Top 3", ""), mrr_text.get("Top 5", ""),
            mrr_text.get("Top 10", ""))

if __name__ == "__main__":
    #假设val_task_id 是这些
    val_task_id = [4, 13, 19, 36, 48]
    # eval_task_name = "_".join([str(i) for i in val_task_id])
    all_task_id = set(range(50))
    th.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # train_task_id = list(all_task_id - set(val_task_id))
    # s3d_model = S3D("../s3d_dict.npy", 512)
    # # s3d = th.compile(s3d)
    # s3d_model = s3d_model.to(device)
    # s3d_model.load_state_dict(th.load("../s3d_howto100m.pth"))
    # s3d_model.eval()
    # (train_video_embeddings_normalized, train_text_embeddings_normalized, validate_video_embeddings_normalized,
    #  validate_text_embeddings_normalized, train_mappings, validate_mappings) = get_s3d_embeddings(train_task_id=train_task_id, val_task_id=val_task_id, s3d=s3d_model, seed=42)
    transform_model = SingleLayerMLP(512, 512).to(device)
    checkpoint_path = '/scr/jzhang96/triplet_text_loss_models/triplet_loss_50_42_s3d_TimeShort_Normtriplet/1555.pth'
    checkpoint = th.load(checkpoint_path)
    transform_model.load_state_dict(checkpoint)
    video_features, text_features, mappings = get_s3d_embeddings_h5(all_task_id)
    print(video_features.shape, text_features.shape)

    mrr_1, mrr_3, mrr_5, mrr_10 = eval_mrr(model=transform_model, evaluate_task=all_task_id,
                                           video_embeddings=video_features.to(device),
                                           text_embeddings=text_features.to(device), mappings=mappings)
    print(mrr_1, mrr_3, mrr_5, mrr_10)

    # variance_threshold = 512
    # pca_text, reduced_train_text = reduce_dimension(text_features.cpu(), variance_threshold,
    #                                                 'text', seed=42, kernel='linear',
    #                                                 val_task_name=val_task_id)  # pca_emb=pca_train_alltext
    # pca_video, reduced_train_video = reduce_dimension(video_features.cpu(), variance_threshold, 'video',
    #                                                   dimension=reduced_train_text.shape[1],
    #                                                   seed=42, kernel='linear',
    #                                                   val_task_name=val_task_id)  # 35，512
    # computed_matrix = compute_M(pca_video.components_, pca_text.components_, variance_threshold, seed=42)
    # with th.no_grad():
    #     transform_model.linear.weight = nn.Parameter(computed_matrix.T.to(device))
    #     transform_model.linear.bias = nn.Parameter(th.zeros(512).to(device))
    #
    # print(th.allclose(transform_model(reduced_train_video.to(device)), normalize_embeddings(th.matmul(reduced_train_video.to(device), computed_matrix.to(device)))))
    # mrr_1, mrr_3, mrr_5, mrr_10 = eval_mrr(model=transform_model, evaluate_task=all_task_id,
    #                                        video_embeddings=reduced_train_video.to(device),
    #                                        text_embeddings=reduced_train_text.to(device), mappings=mappings)
    # print(mrr_1, mrr_3, mrr_5, mrr_10)