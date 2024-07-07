import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding, filter_top_embeddings, SthDataset, OpenXDataset, \
    AugmentationPipeline, Embedding_gpu
from sklearn.model_selection import train_test_split
import torch as th
import h5py
import random
from torch.utils.data import Dataset, DataLoader
from s3dg import S3D
from sklearn.decomposition import PCA
import joblib
import torch.nn.functional as F
from torch.utils.data import Subset
from Wandb_utils import Test_Model, image_aug_fn, AugmentedDataset, normalize_and_pca, save_augmented_videos, \
    reduce_with_pca, MetaDataset, plot_s3d
import numpy as np
import os
from pca import plot_embeddings

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
from mlp import normalize_embeddings, standradize_embeddings
import argparse
import wandb
h5_file_path = "/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5"
h5_file = h5py.File(h5_file_path, 'r')

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
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


def select_top_d_components(embeddings, variance_threshold, d):
    # Perform PCA with variance threshold
    pca = PCA(n_components=variance_threshold)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Get the explained variance ratios from the PCA
    explained_variance_ratios = pca.explained_variance_ratio_

    # Get the indices of the top d explained variance ratios
    top_d_indices = np.argsort(explained_variance_ratios)[-d:][::-1]

    # Select the top d components from the reduced embeddings
    final_reduced_embeddings = reduced_embeddings[:, top_d_indices]

    top_d_components = pca.components_[top_d_indices]

    pca_text = Create_PCA(pca, top_d_components, top_d_indices, d)

    return torch.from_numpy(final_reduced_embeddings).float(), top_d_components, pca_text


def Create_PCA(pca, top_d_components, top_d_indices, d):
    pca_top_d = PCA(n_components=d)
    pca_top_d.components_ = top_d_components
    pca_top_d.explained_variance_ = pca.explained_variance_[top_d_indices]
    pca_top_d.explained_variance_ratio_ = pca.explained_variance_ratio_[top_d_indices]
    pca_top_d.mean_ = pca.mean_
    pca_top_d.n_components_ = d
    pca_top_d.n_features_ = pca.n_features_
    pca_top_d.n_samples_ = pca.n_samples_
    return pca_top_d


def reduce_dimension(
        embeddings, variance_threshold, train_size, embed_type, seed, dimension=None, filter=False
):
    if variance_threshold == 0:
        return None, embeddings.float()

    if dimension:
        pca = PCA(n_components=dimension)
    else:
        pca = PCA(n_components=variance_threshold)
    reduced_embeddings = pca.fit_transform(embeddings)
    if filter:
        model_filename = (
            f"saved_model/M/OpenX/droid/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"saved_model/M/OpenX/droid/pca_model/pca_model_{embed_type}_{variance_threshold}_{train_size}_Seed{seed}.pkl"
        )
    joblib.dump(pca, model_filename)
    print(f"PCA model for {embed_type} saved to {model_filename}")
    return pca, torch.from_numpy(reduced_embeddings).float()


def reduce_dimension_trained(
        embeddings, variance_threshold, train_size, embed_type, filter=False
):
    if filter:
        model_filename = (
            f"saved_model/M/filter/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    else:
        model_filename = (
            f"saved_model/M/OpenX/droid/alignXtoX/pca_model_{embed_type}_{variance_threshold}_{train_size}.pkl"
        )
    pca = joblib.load(model_filename)
    reduced_embeddings = pca.transform(embeddings)

    print(f"Using PCA_{embed_type} from {model_filename}")
    return pca.components_, torch.from_numpy(reduced_embeddings).float()


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


def cos_similarity_score(adjust_video_embeddings, text_embeddings_pca):
    sim_scores = F.cosine_similarity(adjust_video_embeddings, text_embeddings_pca, dim=1)
    return sim_scores


def finetune_M_Random(model, optimizer, reduced_video, reduced_text, path, milnce_loss):
    adjusted_video_embeddings = model(reduced_video)
    loss = milnce_loss(adjusted_video_embeddings, reduced_text)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint model saved to {path}")


def xclip_embeddings(task_id, h5_file):
    video_features = []
    train_task_name = []
    text_features = []
    video_ids = []
    text_labels = []

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

class DoubleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DoubleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    train_video_features = []
    train_task_name = []
    train_text_features = []
    eval_video_features = []
    eval_task_name = []
    eval_text_features = []
    wandb.login(key="894302be844229c43f7c4f673f3f715efc55c3fd")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", type=bool, default=False)
    args = parser.parse_args()
    augmentation = False
    if args.augmentation:
        augmentation = True
    variance_thresholds = [512] #0.9 0.95
    sample_sizes = [45 * 15]  # [1, 2, 4, 8, 16, 21]
    seeds = [42]
    kernel_type = ['rbf'] #‘rbf’
    pca_sample_size = [15]  # 600, 1000, 1500, 2000
    strong_augmented = ['no']  # 'weak', 'strong' , 'weak', 'normal'
    transform_types = ['linear', 'non linear']
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_task_id = [4, 13, 19, 36, 48]#[1, 18, 19, 48, 49]  # [13, 15, 35, 37, 46]#[8, 24, 38, 41, 47] #[4, 13, 19, 36, 48] #[13, 14, 15, 16, 17]#[1, 18, 19, 48, 49] #[23, 24, 25, 26, 40] #[1, 18, 19, 48, 49] #[13, 14, 15, 16, 17] #[4, 13, 19, 36, 48]#[8, 24, 38, 41, 47]
    eval_task_name = "_".join([str(i) for i in val_task_id])
    all_task_id = set(range(50))
    train_task_id = list(all_task_id - set(val_task_id))


    #h5_file.close()

    # print("uni", len(unique_indices))
    for size_multiplier in sample_sizes:
        current_sample_size = size_multiplier
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_video_embeddings, train_text_embeddings, train_mappings = xclip_embeddings(train_task_id, h5_file)
            validate_video_embeddings_normalized, validate_text_embeddings_normalized, validate_mappings = xclip_embeddings(
                val_task_id, h5_file)
            for strong in strong_augmented:
                for num_augmented_per_video in pca_sample_size:
                    for transform_type in transform_types:
                        for kernel in kernel_type:
                            for variance_threshold in variance_thresholds:
                                print(
                                    f"Training with variance threshold {variance_threshold} and sample size {current_sample_size}."
                                )
                                wandb_log = {}
                                wandb.init(
                                    project="p-roboclip-v2",
                                    name=f"XCLIP_PCA-{variance_threshold}_sample-{current_sample_size}_seed-{seed}_{strong}_aug_{kernel}_{transform_type}_{eval_task_name}",
                                    config={
                                        "learning_rate": 0.0005,
                                        #"model": "M from PCA computed",  # random ini, PCA computed, MLP ini
                                        "VLM": "XCLIP",
                                        "PCA kernel": kernel,
                                        "dataset": "metaworld",
                                        "epochs": 2000,
                                        "sample size": size_multiplier,
                                        "seed": seed,
                                        "variance threshold": variance_threshold,
                                        "Unique": True,
                                        "Augmentation": 'no',  # 'Text Augmentation'
                                        "Unseen Scenes": 'Seen tasks 4 13 19 36 48',
                                        # 'Handle(23-26) + Soccer(40)',#'Seen tasks 4 13 19 36 48', #'Seen tasks 13 15 35 37 46' drawer(18-19) + basketball(1) + window(48-49)#'Seen tasks 8 24 38 41 47'
                                        "Sample size used for PCA Matrix": num_augmented_per_video,
                                        "Stronger augmentation": strong,
                                        "Transform type": transform_type
                                    }
                                )
                                reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text = normalize_and_pca(
                                    train_video_embeddings, train_text_embeddings, validate_video_embeddings_normalized,
                                    validate_text_embeddings_normalized, variance_threshold, current_sample_size, seed, device,
                                    strong, num_augmented_per_video, kernel=kernel, val_task_name=eval_task_name)
                                Test_Model(normalize_embeddings(train_video_embeddings).float().to(device),
                                           normalize_embeddings(train_text_embeddings).float().to(device),
                                           train_mappings, -1, None, val_task_id, 'Train', wandb_log)
                                Test_Model(validate_video_embeddings_normalized.float().to(device),
                                           validate_text_embeddings_normalized.float().to(device), validate_mappings, -1, None,
                                           val_task_id, 'Validate', wandb_log)
                                wandb.log(wandb_log)

                                if transform_type == 'linear':
                                    model = nn.Linear(reduced_train_video.shape[1], reduced_train_text.shape[1], bias=False).to(
                                        device)
                                    if variance_threshold != 0 and kernel != 'rbf':
                                        pretrained_matrix = compute_M(pca_video.components_, pca_text.components_,
                                                                      variance_threshold,
                                                                      current_sample_size, seed, filter=False)
                                        print(pretrained_matrix.shape)
                                        model.weight.data = pretrained_matrix.T.to(device)
                                elif transform_type == 'non linear':
                                    model = DoubleMLP(reduced_train_video.shape[1], reduced_train_video.shape[1], reduced_train_text.shape[1]).to(device)
                                # model = SimpleMLP(reduced_train_video.shape[1], reduced_train_text.shape[1]).to(device)



                                milnce_loss = MILNCELoss()
                                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

                                save_dir = (f'/scr/yusenluo/RoboCLIP/visualization/saved_model/M/metaworld_xclip/'
                                            f'{eval_task_name}_Seed_{seed}/{variance_threshold}_Aug_{strong}_{current_sample_size}_{kernel}/{transform_type}')
                                checkpoint_dir = f'{save_dir}/checkpoint'
                                os.makedirs(save_dir, exist_ok=True)
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                final_model_path = f"{save_dir}/MLP_model.pth"
                                for epoch in range(2000):
                                    print(
                                        f"Training with {size_multiplier} samples in Epoch {epoch}, PCA {variance_threshold}, Seed{seed}")
                                    checkpoint_model_path = f"{checkpoint_dir}/MLP_model_Epoch{epoch + 1}.pth"
                                    wandb_log = {}
                                    # sampled_data_loader = DataLoader(
                                    #     train_dataset, batch_size=32, shuffle=True, num_workers=5, pin_memory=True
                                    # )
                                    # augmented_video_embeddings, augmented_text_embeddings, _, augmented_mappings = Embedding_gpu(
                                    #     s3d, sampled_data_loader
                                    # )
                                    # reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text = reduce_with_pca(
                                    #     augmented_video_embeddings, augmented_text_embeddings,
                                    #     validate_video_embeddings_normalized,
                                    #     validate_text_embeddings_normalized, pca_video, pca_text, device)
                                    finetune_M_Random(model, optimizer, reduced_train_video, reduced_train_text,
                                                      checkpoint_model_path,
                                                      milnce_loss)
                                    adjusted_train_video_embeddings, _ = Test_Model(reduced_train_video, reduced_train_text,
                                                                                    train_mappings, epoch, model, val_task_id,
                                                                                    'Train', wandb_log)
                                    adjusted_validate_video_embeddings, _ = Test_Model(reduced_validate_video,
                                                                                       reduced_validate_text,
                                                                                       validate_mappings, epoch, model,
                                                                                       val_task_id, 'Validate', wandb_log)
                                    wandb.log(wandb_log)
                                torch.save(model.state_dict(), final_model_path)
                                print(f"Final model saved to {final_model_path}")
                                # plot_embeddings(adjusted_validate_video_embeddings.cpu(),
                                #                 reduced_validate_text.cpu(), validate_mappings, 'plots',
                                #                 'meta_val.png')
                                # plot_embeddings(adjusted_train_video_embeddings.cpu(),
                                #                 reduced_train_text.cpu(), validate_mappings, 'plots',
                                #                 'meta_train.png')

                                wandb.finish()
