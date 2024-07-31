import torch
from s3dg import S3D
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import argparse
import h5py
from Wandb_utils import normalize_and_pca
from tqdm import tqdm
import imageio
from torch.utils.data import DataLoader, Dataset
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 获取当前文件的目录
current_dir = os.path.dirname(__file__)

# 获取上级目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 获取 losses 目录的路径
losses_dir = os.path.join(parent_dir, 'losses')

# 将 losses 目录添加到 sys.path
sys.path.append(losses_dir)
from dataloader import GifDataset, GifProgressDataset
import torch.nn as nn
import torch as th
import wandb
import random
from Wandb_utils import Test_Model, MetaDataset
from Dataload import Embedding_gpu
text_h5_file_path = "/scr/jzhang96/metaworld_s3d_text.h5"
text_h5_file = h5py.File(text_h5_file_path, 'r')


def compute_M(X_S, X_T, variance_threshold, train_size, seed, filter):
    M = np.dot(X_S, X_T.T)  # 35 35
    M_tensor = torch.from_numpy(M).float()
    if filter:
        save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/filter"
    else:
        save_dir = "/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/alignXtoX_vid_aug"
    os.makedirs(save_dir, exist_ok=True)
    M_model_path = f"{save_dir}/M_model_{variance_threshold}_{train_size}_Seed{seed}.pth"
    torch.save(M_tensor, M_model_path)
    print(f'M model saved to {M_model_path}')
    return M_tensor

class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

        # self.linear_2 = nn.Linear(output_dim, output_dim)
        # self.linear_3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        # x = F.relu(x)
        # x = self.linear_2(x)
        # x = F.relu(x)
        # x = self.linear_3(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)


def find_hard_negative(video_embeddings, text_embeddings, mappings, batch_size):
    triplets = []
    max_sims = []
    min_sims = []
    mean_sims = []

    current_video_embeds = video_embeddings
    current_text_embeds = text_embeddings
    current_video_ids = list(mappings['index_to_video_id'].values())
    current_text_labels = list(mappings['index_to_text_label'].values())

    for j, text_emb in enumerate(current_text_embeds):
        gt_text_emb = text_emb
        text_label = current_text_labels[j]

        # Compute cosine similarity with all video embeddings
        similarity_scores = F.cosine_similarity(gt_text_emb.unsqueeze(0), video_embeddings, dim=1)

        # Exclude the current positive sample and same text label
        similarity_scores[j] = -1.0  # Ignore current positive sample
        for k, lbl in enumerate(current_text_labels):
            if lbl == text_label:
                similarity_scores[k] = -1.0  # Ignore same text label

        # Collect similarity statistics
        valid_scores = similarity_scores[similarity_scores != -1.0]
        max_sims.append(valid_scores.max().item())
        min_sims.append(valid_scores.min().item())
        mean_sims.append(valid_scores.mean().item())

        # Find the hardest negative sample
        neg_idx = valid_scores.argmax().item()
        neg_video_emb = video_embeddings[neg_idx]

        pos_video_emb = current_video_embeds[j]

        triplets.append((gt_text_emb, pos_video_emb, neg_video_emb, torch.tensor(1, dtype=torch.int32),  # Ensure type is Tensor
                torch.tensor(0, dtype=torch.float32))) # neg type, progress

    triplet_dataset = TripletDataset(triplets)
    avg_max_simi = sum(max_sims) / len(max_sims)
    avg_min_simi = sum(min_sims) / len(min_sims)
    avg_mean_simi = sum(mean_sims) / len(mean_sims)

    return triplet_dataset, avg_max_simi, avg_min_simi, avg_mean_simi



def triplet_loss(gt, positive, negative, type, margin = (0.9, 0.75, 0.7, 0.1), progress = None):
    #

    pos_sim = cosine_similarity(gt, positive)
    neg_sim = cosine_similarity(gt, negative)

    loss = torch.zeros_like(pos_sim).cuda()

    # type1: hard negative margin = 0.9
    mask_type_1 = (type == 1)
    loss[mask_type_1] = F.relu(margin[0] - pos_sim[mask_type_1] + neg_sim[mask_type_1])

    # type2: semi-hard negative margin = 0.75
    mask_type_2 = (type == 2)
    loss[mask_type_2] = F.relu(margin[1] - pos_sim[mask_type_2] + neg_sim[mask_type_2])

    # type3: adaptive margin
    mask_type_3 = (type == 3)
    progress = progress[mask_type_3]
    # adaptive margin range from 0.7 to 0.1
    adaptive_margin = (margin[2] + (margin[3] - margin[2]) * progress).cuda()
    loss[mask_type_3] = F.relu(adaptive_margin - pos_sim[mask_type_3] + neg_sim[mask_type_3])

    return loss.mean()


def triplet_loss_lower_bound(gt, positive, negative, type, margin=(0.1, 0.2, 0.3, 0.9),
                             progress=None):  # may need a lower bound for progress
    # for progress, we need to have an lower bound

    pos_sim = cosine_similarity(gt, positive)
    neg_sim = cosine_similarity(gt, negative)

    loss = torch.zeros_like(pos_sim).cuda()

    # type1: hard negative margin = 0.1
    mask_type_1 = (type == 1)
    loss[mask_type_1] = F.relu(margin[0] - pos_sim[mask_type_1] + neg_sim[mask_type_1])

    # type2: semi-hard negative margin = 0.2
    mask_type_2 = (type == 2)
    loss[mask_type_2] = F.relu(margin[1] - pos_sim[mask_type_2] + neg_sim[mask_type_2])

    # type3: adaptive margin # can change to L1 loss
    mask_type_3 = (type == 3)
    progress = progress[mask_type_3]
    # adaptive margin range from 0.3 to 0.9
    # adaptive_margin_upper_bound = (margin[2] + (margin[3] - margin[2]) * progress).cuda()
    # adaptive_margin_lower_bound = (margin[2] + (margin[3] - margin[2]) * progress - 0.1).cuda()
    # #make sure negative similarity - positive similarity is between upper and lower bound
    # simi_diff = neg_sim[mask_type_3] - pos_sim[mask_type_3]
    # loss_upper = F.relu(adaptive_margin_upper_bound - simi_diff)
    # loss_lower = F.relu(simi_diff - adaptive_margin_lower_bound)
    # loss[mask_type_3] = loss_upper + loss_lower
    progress_range = ((margin[3] - margin[2]) * progress + margin[2]).cuda()
    loss[mask_type_3] = F.l1_loss(neg_sim[mask_type_3] - pos_sim[mask_type_3], progress_range)

    return loss.mean()


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()


class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        ground_truth, positive, negative, neg_type, progress = self.triplets[idx]
        return {
            'ground_truth': ground_truth,
            'positive': positive,
            'negative': negative,
            "type": neg_type.clone().detach(),
            "progress": progress.clone().detach()
        }





class Text_Cond_Dataset(Dataset):
    def __init__(self, args, task_ids=None):
        self.h5_file = h5py.File(args.h5_path, "r")
        text_h5_file_path = "/scr/jzhang96/metaworld_s3d_text.h5"
        self.text_h5_file = h5py.File(text_h5_file_path, 'r')
        self.keys = list(self.h5_file.keys())
        # self.keys = ['door-close-v2-goal-hidden']
        all_keys = list(self.h5_file.keys())

        # 仅保留指定的任务ID对应的keys
        if task_ids is not None:
            self.keys = [key for i, key in enumerate(all_keys) if i in task_ids]
        else:
            self.keys = all_keys

        self.vlm_name = args.model_name
        # if self.vlm_name == "xclip":
        # self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
        # self.model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").cuda()
        # else:
        #     self.s3d_model = load_model("s3d")

        self.shuffle_time = args.time_shuffle
        self.shorten_time = args.time_shorten
        self.candidate_type = []
        if args.time_shuffle:
            self.candidate_type.append(2)
        if args.time_shorten:
            self.candidate_type.append(3)
    def __len__(self):
        return len(self.keys) * 30

    def __getitem__(self, idx):
        real_idx = idx % len(self.keys)  # env name
        key = self.keys[real_idx]
        group = self.h5_file[key]
        text_group = self.text_h5_file[key]
        gif_names = list(group.keys())
        # print(gif_names)
        # sample gt sample
        gt_gif_name = random.choice(gif_names)
        gt_array = group[gt_gif_name][()]

        gt_text = th.from_numpy(np.array(text_group["embedding"]))
        # print(gt_text.shape)
        # print(f"key: {key}")

        # sample positive sample
        pos_gif_name = random.choice(gif_names)  # 同task的
        pos_array = group[pos_gif_name][()]

        # choose shuffle time or sample negative sample from other key, choose 1 from 2
        neg_type = random.choice(self.candidate_type)
        progress = 0
        if neg_type == 1:
            neg_array = self.sample_negative_func(key)
        elif neg_type == 2:
            neg_array = self.shuffle_time_func(gt_array.copy())
        elif neg_type == 3:
            neg_array, progress = self.shorten_time_func(gt_array.copy())

        # sample frames
        gt_array = self.sample_frames(gt_array)
        pos_array = self.sample_frames(pos_array)
        neg_array = self.sample_frames(neg_array)

        # preprocess
        if self.vlm_name == "xclip":

            gt_array = self.preprocess_xclip(gt_array)
            pos_array = self.preprocess_xclip(pos_array)
            neg_array = self.preprocess_xclip(neg_array)

        else:
            gt_array = gt_array / 255
            pos_array = pos_array / 255
            neg_array = neg_array / 255

            gt_array = gt_array.transpose(3, 0, 1, 2)
            pos_array = pos_array.transpose(3, 0, 1, 2)
            neg_array = neg_array.transpose(3, 0, 1, 2)

            gt_array = th.from_numpy(gt_array).float()
            pos_array = th.from_numpy(pos_array).float()
            neg_array = th.from_numpy(neg_array).float()

        output_dict = {
            "text_embedding": gt_text,
            "gt_array": gt_array,
            "pos_array": pos_array,
            "neg_array": neg_array,
            "type": neg_type,
            "progress": progress
        }

        return output_dict

    def shuffle_time_func(self, array):
        random_index = np.random.permutation(len(array))
        return array[random_index]

    def shorten_time_func(self, array):
        video_length = len(array)
        progress = 1
        if len(array) > 33:
            max_len = min(32, len(array) - 1)
            # random choose end from 32, max_len
            end = random.randint(32, max_len)
            array = array[:end]

            progress = end / video_length

        return array, progress

    def sample_negative_func(self, key):
        other_key = key
        while other_key == key:
            other_key = random.choice(self.keys)
        other_group = self.h5_file[other_key]
        other_gif_names = list(other_group.keys())
        neg_gif_name = random.choice(other_gif_names)
        neg_array = other_group[neg_gif_name][()]
        return neg_array

    def sample_frames(self, array, num_frames=32):
        if len(array) > num_frames:
            indices = np.linspace(0, len(array) - 1, num_frames, dtype=int)
            return array[indices]
        else:
            more_frames = num_frames - len(array)
            last_frame = array[-1:]
            for i in range(more_frames):
                array = np.concatenate([array, last_frame], axis=0)
        return array

    def preprocess_xclip(self, array):
        # crop to from 250x250 to 224x224
        # if array.shape != (32, 250, 250, 3):

        array = array[:, 13:237, 13:237, :]
        pixel_values = self.processor(videos=list(array), return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values

    def __del__(self):
        self.h5_file.close()


def process_video_features(pos_array, neg_array, model, transform_model, model_name="s3d", batch_size=32):
    # 将正样本和负样本连接在一起
    samples = torch.cat([pos_array, neg_array]).cuda()

    # 初始化一个空的列表来存储处理后的特征
    video_features_list = []

    # 手动将样本分成小批次进行处理
    for i in range(0, samples.shape[0], batch_size):
        batch_samples = samples[i:i + batch_size]

        with th.no_grad():
            if model_name == "xclip":
                pixel_values = xclip_processor.processor(videos=list(batch_samples),
                                                         return_tensors="pt").pixel_values.squeeze(0)
                video_features = model.get_video_features(pixel_values.cuda())
            else:
                video_features = model(batch_samples)["video_embedding"]

            video_features = normalize_embeddings(video_features)
            video_features = transform_model(video_features)
            video_features_list.append(video_features)

    # 将所有小批次的特征连接在一起
    video_features = torch.cat(video_features_list, dim=0)

    # 获取正样本和负样本的特征
    batch_size = neg_array.shape[0]
    pos_features = video_features[:batch_size]
    neg_features = video_features[batch_size:2 * batch_size]

    return pos_features, neg_features


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    h5_file = h5py.File(args.h5_path, "r")
    model_name = args.model_name

    val_task_id = args.val_task_ids
    # [13, 15, 35, 37, 46]#[8, 24, 38, 41, 47] #[4, 13, 19, 36, 48] #[13, 14, 15, 16, 17]#[1, 18, 19, 48, 49] #[23, 24, 25, 26, 40] #[1, 18, 19, 48, 49] #[13, 14, 15, 16, 17] #[4, 13, 19, 36, 48]#[8, 24, 38, 41, 47]
    all_task_id = set(range(50))
    eval_task_name = "_".join([str(i) for i in val_task_id])
    train_task_id = list(all_task_id - set(val_task_id))

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "p-roboclip-v2"
    # wandb_eval_task_name = "_".join([str(i) for i in eval_tasks])
    # experiment_name = args.experiment_name + "_" + wandb_eval_task_name + "_" + str(args.seed)
    # if args.mse:
    #     experiment_name = experiment_name + "_mse_" + str(args.mse_weight)
    experiment_name = "triplet_loss" + "_" + str(args.seed) + "_" + args.model_name + "_l1" + "_" + str(args.margin) + f"_bs_{args.batch_size}"
    if args.time_shuffle:
        experiment_name += "_TimeShuffle"
    if args.time_shorten:
        experiment_name += "_TimeShort"
    if args.norm:
        experiment_name += "_Norm"
    if args.add_lower_bound:
        experiment_name += "_LowerBound"

    experiment_name += f"_text_{eval_task_name}"

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="adaptive_triplet_loss_training",
        config=args,
        name=experiment_name,
    )


    if model_name == "xclip":
        xclip_net = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").cuda()
        xclip_net.eval()
        # pixel_values = self.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)
        xclip_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
    else:
        s3d_model = S3D('../s3d_dict.npy', 512)
        s3d_model.load_state_dict(th.load('../s3d_howto100m.pth'))
        s3d_model.eval().cuda()


    train_dataset = MetaDataset(
        "/scr/yusenluo/RoboCLIP/metaworld_generate_gifs", train_task_id, num_samples=15, seed=args.seed
    )
    val_dataset = MetaDataset(
        "/scr/yusenluo/RoboCLIP/metaworld_generate_gifs", val_task_id, num_samples=15, seed=args.seed
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=15, shuffle=False, num_workers=5
    )
    validate_data_loader = DataLoader(
        val_dataset, batch_size=15, shuffle=False, num_workers=5
    )
    train_video_embeddings, train_text_embeddings, _, train_mappings = Embedding_gpu(
        s3d_model, train_data_loader
    )
    validate_video_embeddings, validate_text_embeddings, _, validate_mappings = Embedding_gpu(
        s3d_model, validate_data_loader
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
    wandb_log = {}
    Test_Model(train_video_embeddings_normalized, train_text_embeddings_normalized, train_mappings, -1,
               None,
               train_task_id, 'Train', wandb_log)
    Test_Model(validate_video_embeddings_normalized,
               validate_text_embeddings_normalized,
               validate_mappings, -1, None,
               val_task_id, 'Validate', wandb_log)
    wandb.log(wandb_log)

    reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text = normalize_and_pca(
        train_video_embeddings_normalized, train_text_embeddings_normalized, validate_video_embeddings_normalized,
        validate_text_embeddings_normalized, variance_threshold=args.pca_variance, current_sample_size=675, seed=args.seed, device=device,
        strong='no', pca_sample_size=15, kernel='linear', val_task_name=eval_task_name)
    transform_model = SingleLayerMLP(512, 512, normalize=args.norm).cuda()
    if args.pca_variance != 0:
        pretrained_matrix = compute_M(pca_video.components_, pca_text.components_,
                                      variance_threshold=args.pca_variance,
                                      train_size=675, seed=args.seed, filter=False)
        transform_model.linear.weight.data = pretrained_matrix.T.to(device)

    # type 2 and 3:
    if args.time_shuffle or args.time_shorten:
        ProgressDataset = Text_Cond_Dataset(args, train_task_id)
        progress_dataloader = DataLoader(ProgressDataset, batch_size=2 * args.batch_size, shuffle=True, num_workers=args.num_workers)
        progress_iter = iter(progress_dataloader)

    # loss_func = nn.TripletMarginLoss(margin=0.5, p=1)
    optimizer = th.optim.Adam(transform_model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        # for i, batch in enumerate(tqdm(dataloader)):
        #     wandb_log = {}
        #     gt_array = batch["gt_array"]
        #     gt_text = batch["text_embedding"].cuda()
        #     pos_array = batch["pos_array"]
        #     neg_array = batch["neg_array"]
        #     batch_size = neg_array.shape[0]
        #     samples = torch.cat([gt_array, pos_array, neg_array]).cuda()
        #     type = batch["type"]
        #     progress = batch["progress"].float().cuda()
        #
        #     with th.no_grad():
        #         if model_name == "xclip":
        #             import pdb;
        #             pdb.set_trace()
        #             pixel_values = xclip_processor.processor(videos=list(array),
        #                                                      return_tensors="pt").pixel_values.squeeze(0)
        #
        #             video_features = xclip_net.get_video_features(samples)
        #
        #         else:
        #             video_features = s3d_model(samples)["video_embedding"]
        #
        #     video_features = normalize_embeddings(video_features)
        #     video_features = transform_model(video_features)
        #
        #     #gt_features = video_features[:batch_size]
        #     gt_features = normalize_embeddings(gt_text)
        #     pos_features = video_features[batch_size:2 * batch_size]
        #     neg_features = video_features[2 * batch_size:]
        for i in range(0, len(reduced_train_video), args.batch_size):
            video_embeddings_batch = reduced_train_video[i:i + args.batch_size]
            text_embeddings_batch = reduced_train_text[i:i + args.batch_size]
            video_embeddings_normalized = normalize_embeddings(transform_model(video_embeddings_batch))
            text_embeddings_normalized = normalize_embeddings(text_embeddings_batch)

            current_mappings = {
                'index_to_video_id': {k: v for k, v in enumerate(range(i, i + len(video_embeddings_batch)))},
                'index_to_text_label': {k: train_mappings['index_to_text_label'][v] for k, v in
                                        enumerate(range(i, i + len(text_embeddings_batch)))}
            }
            #type 1: other key
            triplet_dataset, avg_max_sim, avg_min_sim, avg_mean_sim = find_hard_negative(video_embeddings_normalized, text_embeddings_normalized,
                                                 current_mappings, batch_size=args.batch_size)
            try:
                progress_batch = next(progress_iter)
            except StopIteration:
                progress_iter = iter(progress_dataloader)
                progress_batch = next(progress_iter)
            gt_array = progress_batch["gt_array"]
            gt_text = progress_batch["text_embedding"].to(device)
            pos_array = progress_batch["pos_array"].to(device)
            neg_array = progress_batch["neg_array"].to(device)
            batch_size = neg_array.shape[0]
            type = progress_batch["type"]
            progress = progress_batch["progress"].float()

            pos_features, neg_features = process_video_features(pos_array, neg_array, s3d_model, transform_model, model_name, batch_size=32)
            #gt_features = video_features[:batch_size]
            gt_features = normalize_embeddings(gt_text)

            for j in range(len(gt_text)):
                triplet_dataset.triplets.append(
                    (gt_features[j], pos_features[j], neg_features[j], torch.tensor(type[j], dtype=torch.int32),  # Ensure type is Tensor
                torch.tensor(progress[j], dtype=torch.float32)))
            print(len(triplet_dataset))
            triplet_loader = DataLoader(triplet_dataset, batch_size=3 * args.batch_size, shuffle=True)

            for triplet_batch in triplet_loader:
                wandb_log = {}
                gt_features = triplet_batch['ground_truth'].to(device)
                pos_features = triplet_batch['positive'].to(device)
                neg_features = triplet_batch['negative'].to(device)
                sample_type = triplet_batch['type'].to(device)
                progress = triplet_batch['progress'].float().to(device)

                if args.add_lower_bound:
                    loss = triplet_loss_lower_bound(gt_features, pos_features, neg_features, type=sample_type, progress=progress)
                else:
                    loss = triplet_loss(gt_features, pos_features, neg_features, type=sample_type, progress=progress)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                Test_Model(reduced_train_video.to(device), reduced_train_text.to(device),
                           train_mappings, epoch, transform_model, train_task_id, 'Train', wandb_log)
                Test_Model(reduced_validate_video.to(device),
                           reduced_validate_text.to(device), validate_mappings, epoch, transform_model,
                           val_task_id, 'Validate', wandb_log)

                wandb_log[f'Train_triplet_loss'] = loss.item()
                wandb_log[f'max_cos_neg_similarity'] = avg_max_sim
                wandb_log[f'mean_cos_neg_similarity'] = avg_mean_sim
                wandb_log[f'min_cos_neg_similarity'] = avg_min_sim
                wandb.log(wandb_log)

        if epoch % 1 == 0:
            if args.model_name == "xclip":
                if not os.path.exists(f"/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/{experiment_name}"):
                    os.makedirs(f"/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(),
                        f"/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/{experiment_name}/{epoch}.pth")
            else:
                if not os.path.exists(f"/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/{experiment_name}"):
                    os.makedirs(f"/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(),
                        f"/scr/yusenluo/RoboCLIP/visualization/saved_model/triplet_loss_models/{experiment_name}/{epoch}.pth")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='xclip', choices=['xclip', 's3d'])
    argparser.add_argument('--time_shuffle', action='store_true')#不需要
    argparser.add_argument('--h5_path', type=str, default='/scr/jzhang96/metaworld_gifs_1.h5')
    argparser.add_argument('--time_shorten', action='store_true')#不需要
    argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--epochs', type=int, default=1000)
    argparser.add_argument('--pca_variance', type=int, default=512)
    argparser.add_argument('--margin', type=float, default=0.9)
    argparser.add_argument('--add_lower_bound', action='store_true')
    argparser.add_argument('--val_task_ids', type=int, nargs='+', default=[4, 13, 19, 36, 48],
                        help="List of task IDs for validation")
    args = argparser.parse_args()
    main(args)