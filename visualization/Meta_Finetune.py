import random

import torch
import torch.nn as nn
from Dataload import VideoTextDataset, list_webm_files, Embedding, filter_top_embeddings, SthDataset, OpenXDataset, \
    AugmentationPipeline, Embedding_gpu
from sklearn.model_selection import train_test_split
import torch as th
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

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
from mlp import normalize_embeddings, standradize_embeddings
import argparse
import wandb


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
        nominator = x * th.eye(x.shape[0])[:, :, None].cuda()
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


def finetune_M_Random(model, optimizer, reduced_video, reduced_text, path, milnce_loss, epoch):
    adjusted_video_embeddings = model(reduced_video)
    video_norm = th.mean(th.norm(adjusted_video_embeddings, dim=1))
    text_norm = th.mean(th.norm(reduced_text, dim=1))
    print(f"adjusted_video_embeddings 的范数是 : {video_norm}")
    print(f"reduced_text 的范数是 : {text_norm}")
    #loss = milnce_loss(adjusted_video_embeddings, reduced_text)
    loss = milnce_loss(reduced_text, adjusted_video_embeddings)
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


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    wandb.login(key="894302be844229c43f7c4f673f3f715efc55c3fd")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", type=bool, default=False)
    variance_threshold = args.pca_variance
    sample_sizes = [45 * 15]  # [1, 2, 4, 8, 16, 21]
    current_sample_size = 675
    seed = args.seed
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s3d = S3D("../s3d_dict.npy", 512)
    # s3d = th.compile(s3d)
    s3d = s3d.to(device)
    s3d.load_state_dict(torch.load("../s3d_howto100m.pth"))
    s3d.eval()

    val_task_id = args.val_task_ids
    all_task_id = set(range(50))
    # train_task_id = list(all_task_id - set(val_task_id))
    train_task_id = all_task_id
    train_dataset = MetaDataset(
        "/scr/yusenluo/RoboCLIP/metaworld_generate_gifs", train_task_id, num_samples=15, seed=args.seed
    )
    val_dataset = MetaDataset(
        "/scr/yusenluo/RoboCLIP/metaworld_generate_gifs", val_task_id, num_samples=15, seed=args.seed
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=2
    )
    validate_data_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=2
    )
    print(len(train_dataset))
    print(len(val_dataset))
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
    # print("uni", len(unique_indices))
    print(
        f"Training with variance threshold {variance_threshold} and sample size {current_sample_size}."
    )
    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "p-roboclip-v2"
    eval_task_name = "_".join([str(i) for i in val_task_id])
    experiment_name = ("milnce_loss" + "_" + str(args.seed) + "_" + args.model_name + f"_PCA_{args.pca_variance}"
                       "_norm" + f"_text_{eval_task_name}" + "_supervised")
    wandb_log = {}
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="milnce_loss_training",
        config=args,
        name=experiment_name,
    )
    reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text = normalize_and_pca(
        train_video_embeddings, train_text_embeddings, validate_video_embeddings_normalized,
        validate_text_embeddings_normalized, variance_threshold, current_sample_size, seed, device,
        strong='no', pca_sample_size=15, kernel='linear', val_task_name=val_task_id)
    Test_Model(normalize_embeddings(train_video_embeddings).float().to(device),
               normalize_embeddings(train_text_embeddings).float().to(device),
               train_mappings, -1, None, val_task_id, 'Train', wandb_log)
    Test_Model(validate_video_embeddings_normalized.float().to(device),
               validate_text_embeddings_normalized.float().to(device), validate_mappings, -1, None,
               val_task_id, 'Validate', wandb_log)
    wandb.log(wandb_log)

    # model = nn.Linear(reduced_train_video.shape[1], reduced_train_text.shape[1], bias=False).to(
    #     device)
    model = SingleLayerMLP(reduced_train_video.shape[1], reduced_train_text.shape[1], normalize=args.norm).to(device)
    # model = SimpleMLP(reduced_train_video.shape[1], reduced_train_text.shape[1]).to(device)
    if variance_threshold != 0:
        pretrained_matrix = compute_M(pca_video.components_, pca_text.components_, variance_threshold,
                                  current_sample_size, seed, filter=False)
        print(pretrained_matrix.shape)
        model.weight.data = pretrained_matrix.T.to(device)

    milnce_loss = MILNCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    save_dir = f'/scr/yusenluo/RoboCLIP/visualization/saved_model/milnce_loss_models/{experiment_name}'
    checkpoint_dir = f'{save_dir}/checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_model_path = f"{save_dir}/{experiment_name}_Final_model.pth"
    for epoch in range(args.epochs):
        print(
            f"Training with {current_sample_size} samples in Epoch {epoch}, PCA {variance_threshold}, Seed{seed}")
        checkpoint_model_path = f"{checkpoint_dir}/M_model_{variance_threshold}_{current_sample_size}_Seed{seed}_Epoch{epoch + 1}_milnce.pth"
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
                          milnce_loss, epoch)
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='xclip', choices=['xclip', 's3d'])
    argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--epochs', type=int, default=1000)
    argparser.add_argument('--pca_variance', type=int, default=512)
    argparser.add_argument('--val_task_ids', type=int, nargs='+', default=[4, 13, 19, 36, 48],
                           help="List of task IDs for validation")
    args = argparser.parse_args()
    main(args)
    #mp.set_start_method('spawn', force=True)

