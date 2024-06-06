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
    reduce_with_pca, Augmented_Batched_Dataset
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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


def eval_M(video_embeddings_pca, M_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    M_model = nn.Linear(video_embeddings_pca.shape[1], video_embeddings_pca.shape[1], bias=False).to(device)
    M_model.load_state_dict(torch.load(M_path))
    M_model.eval()
    # Matrix = torch.load(M_path).to(device)
    with torch.no_grad():
        # adjust_video_embeddings = torch.matmul(video_embeddings_pca, Matrix)
        adjust_video_embeddings = M_model(video_embeddings_pca)
    return adjust_video_embeddings


def eval_MLP(video_embeddings, M_path):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = SimpleMLP(video_embeddings.shape[1], video_embeddings.shape[1]).to(device)
    model.load_state_dict(torch.load(M_path))
    with torch.no_grad():
        adjust_video_embeddings = model(video_embeddings)
    return adjust_video_embeddings


def cos_similarity_score(adjust_video_embeddings, text_embeddings_pca):
    sim_scores = F.cosine_similarity(adjust_video_embeddings, text_embeddings_pca, dim=1)
    return sim_scores


def finetune_M(model, optimizer, reduced_video, reduced_text, path, milnce_loss):
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


def finetune_MLP(num_epochs, video_embeddings, text_embeddings, variance_threshold, current_sample_size):
    video_embeddings = normalize_embeddings(video_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)
    save_dir = '/scr/yusenluo/RoboCLIP/visualization/saved_model/OpenX/droid/mlp_model'
    milnce_loss = MILNCELoss()

    X_T, reduced_text = reduce_dimension_trained(text_embeddings, variance_threshold, current_sample_size, 'text',
                                                 filter=filter)
    X_S, reduced_video = reduce_dimension_trained(video_embeddings, variance_threshold,
                                                  current_sample_size, 'video', filter=filter)
    model = SimpleMLP(reduced_video.shape[1], reduced_text.shape[1]).to(device)
    reduced_text = reduced_text.to(device)
    reduced_video = reduced_video.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epochs):
        adjusted_video_embeddings = model(reduced_video)
        # similarity_matrix = reduced_text @ adjusted_video_embeddings.T
        # diagonal_similarities = torch.diag(similarity_matrix)
        # loss = -torch.mean(diagonal_similarities)
        loss = milnce_loss(adjusted_video_embeddings, reduced_text)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    final_model_path = f"{save_dir}/MLP_model_{variance_threshold}_{current_sample_size}_milnce.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


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


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    wandb.login(key="894302be844229c43f7c4f673f3f715efc55c3fd")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", type=bool, default=False)
    args = parser.parse_args()
    augmentation = False
    if args.augmentation:
        augmentation = True
    variance_thresholds = [512]
    sample_sizes = [10]  # [1, 2, 4, 8, 16, 21]
    seeds = [4]
    pca_sample_size = [59, 99, 199] #600, 1000, 1500, 2000
    strong_augmented = ['strong'] #'weak', 'strong' , 'weak', 'normal'
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths = list_webm_files(
        "../20bn-something-something-v2/train"
    )  # '../20bn-something-something-v2'
    # print(len(video_paths))
    s3d = S3D("../s3d_dict.npy", 512)
    # s3d = th.compile(s3d)
    s3d = s3d.cuda()
    s3d.load_state_dict(torch.load("../s3d_howto100m.pth"))
    s3d.eval()

    video_text_dataset = OpenXDataset(
        '/scr/yusenluo/RoboCLIP/OpenX/droid', random_samples=False, dataset_name='droid'
    )
    seen_labels = set()
    unique_indices = []
    for idx in range(len(video_text_dataset)):
        item = video_text_dataset[idx]
        text_label = item['text']
        if text_label not in seen_labels:
            seen_labels.add(text_label)
            unique_indices.append(idx)

    unique_dataset = Subset(video_text_dataset, unique_indices)
    train_dataset, val_dataset = train_test_split(unique_dataset, test_size=0.2, random_state=42)
    print(len(train_dataset))
    train_data_loader = DataLoader(
        train_dataset, batch_size=50, shuffle=False, num_workers=5
    )
    validate_data_loader = DataLoader(
        val_dataset, batch_size=50, shuffle=False, num_workers=5
    )
    # train_video_embeddings, train_text_embeddings, embeddings_dataset, train_mappings = Embedding(
    #     s3d, train_data_loader
    # )
    validate_video_embeddings, validate_text_embeddings, embeddings_dataset, validate_mappings = Embedding_gpu(
        s3d, validate_data_loader
    )
    validate_video_embeddings_normalized = normalize_embeddings(
        validate_video_embeddings
    ).clone()
    validate_text_embeddings_normalized = normalize_embeddings(
        validate_text_embeddings
    ).clone()
    # print("uni", len(unique_indices))
    for size_multiplier in sample_sizes:
        current_sample_size = size_multiplier
        for seed in seeds:
            for strong in strong_augmented:
                for num_augmented_per_video in pca_sample_size:
                    torch.manual_seed(seed)
                    indices = torch.randperm(len(train_dataset))[:size_multiplier]
        
                    # sampled_video_embeddings = train_video_embeddings[indices]
                    # sampled_text_embeddings = train_text_embeddings[indices]
        
                    # if augmentation:
                    augmentation_pipeline = AugmentationPipeline(device, strong)
                    sampled_dataset = Subset(train_dataset, indices)
        
                    #num_augmented_per_video = 99
                    augmented_dataset = AugmentedDataset(sampled_dataset, augmentation_pipeline, device=device,
                                                             num_augmented_samples=99)
                    augmented_dataset_pca = AugmentedDataset(sampled_dataset, augmentation_pipeline, device=device,
                                                         num_augmented_samples=num_augmented_per_video)
                    save_augmented_videos(augmented_dataset_pca, f'/scr/yusenluo/RoboCLIP/OpenX/droid_augmented_video_{strong}/Seed_{seed}',
                                          num_videos=15)
                    print(len(augmented_dataset_pca))
        
                    pca_data_loader = DataLoader(
                        augmented_dataset_pca, batch_size=32, shuffle=True, num_workers=5, pin_memory=True
                    )
                    augmented_video_embeddings, augmented_text_embeddings, _, mappings = Embedding_gpu(
                        s3d, pca_data_loader
                    )
        
                    for variance_threshold in variance_thresholds:
                        print(
                            f"Training with variance threshold {variance_threshold} and sample size {current_sample_size}."
                        )
                        wandb.init(
                            project="p-roboclip-v2",
                            name=f"PCA-{variance_threshold}_sample-{current_sample_size}_seed-{seed}_Computed_Ini_Vid_Aug_{strong}_{num_augmented_per_video}",
                            config={
                                "learning_rate": 0.0005,
                                "model": "M from PCA computed",  # random ini, PCA computed, MLP ini
                                "dataset": "droid",
                                "epochs": 250,
                                "sample size": size_multiplier,
                                "seed": seed,
                                "variance threshold": variance_threshold,
                                "Unique": True,
                                "Augmentation": 'Video Augmentation 1',
                                "Test": 'Sample size used for PCA Matrix AND different augmentation',
                                "Sample size used for PCA Matrix": num_augmented_per_video,
                                "Stronger augmentation": strong,
                            }
                        )
                        reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text, pca_video, pca_text = normalize_and_pca(
                            augmented_video_embeddings, augmented_text_embeddings, validate_video_embeddings_normalized,
                            validate_text_embeddings_normalized, variance_threshold, current_sample_size, seed, device, strong, num_augmented_per_video)
                        Test_Model(augmented_video_embeddings.float().to(device), augmented_text_embeddings.float().to(device),
                                   mappings, -1, None, 'Train')
                        Test_Model(validate_video_embeddings_normalized.float().to(device),
                                   validate_text_embeddings_normalized.float().to(device), validate_mappings, -1, None,
                                   'Validate')
        
                        model = nn.Linear(reduced_train_video.shape[1], reduced_train_text.shape[1], bias=False).to(device)
                        # model = SimpleMLP(reduced_train_video.shape[1], reduced_train_text.shape[1]).to(device)
        
                        pretrained_matrix = compute_M(pca_video.components_, pca_text.components_, variance_threshold,
                                                      current_sample_size, seed, filter=False)
                        print(pretrained_matrix.shape)
                        model.weight.data = pretrained_matrix.T.to(device)
        
                        milnce_loss = MILNCELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
                        save_dir = f'/scr/yusenluo/RoboCLIP/visualization/saved_model/M/OpenX/droid/video_aug_finetune_M_{strong}_pca_matrix_{num_augmented_per_video}'
                        checkpoint_dir = f'{save_dir}/checkpoint_{seed}'
                        os.makedirs(save_dir, exist_ok=True)
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        final_model_path = f"{save_dir}/M_model_{variance_threshold}_{current_sample_size}_Seed{seed}_milnce.pth"
                        for epoch in range(300):
                            print(
                                f"Training with {size_multiplier} samples in Epoch {epoch}, PCA {variance_threshold}, Seed{seed}")
                            checkpoint_model_path = f"{checkpoint_dir}/M_model_{variance_threshold}_{current_sample_size}_Seed{seed}_Epoch{epoch + 1}_milnce.pth"
                            # augmented_dataset = Augmented_Batched_Dataset(sampled_dataset, augmentation_pipeline,
                            #                                               num_augmented_samples=99)
                            sampled_data_loader = DataLoader(
                                augmented_dataset, batch_size=32, shuffle=True, num_workers=5, pin_memory=True
                            )
                            augmented_video_embeddings, augmented_text_embeddings, _, augmented_mappings = Embedding_gpu(
                                s3d, sampled_data_loader
                            )
                            reduced_train_video, reduced_train_text, reduced_validate_video, reduced_validate_text = reduce_with_pca(
                                augmented_video_embeddings, augmented_text_embeddings, validate_video_embeddings_normalized,
                                validate_text_embeddings_normalized, pca_video, pca_text, device)
                            finetune_M_Random(model, optimizer, reduced_train_video, reduced_train_text, checkpoint_model_path,
                                              milnce_loss)
                            Test_Model(reduced_train_video, reduced_train_text, augmented_mappings, epoch, model, 'Train')
                            Test_Model(reduced_validate_video, reduced_validate_text, validate_mappings, epoch, model,
                                       'Validate')
                        torch.save(model.state_dict(), final_model_path)
                        print(f"Final model saved to {final_model_path}")
                        wandb.finish()
