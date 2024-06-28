import os
import sys
# add the "../" directory to the sys.path
parent_dir_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
parent_dir_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xclip/'))
sys.path.append(parent_dir_1)
sys.path.append(parent_dir_2)
import cv2
import random
from tqdm import tqdm
# import xclip
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
import torch.nn.functional as F
import torch
from transform_utils import learn_pca_model, transform_embeddings, compute_M, get_model, MILNCELoss
import wandb
from transform_utils import compute_similarity_sorted, plot_scatter_text
import joblib

# eval_tasks = [23,24,25,26,40]
eval_tasks = [4,13,19,36,48]
total_tasks = [i for i in range(50)]
train_tasks = [i for i in total_tasks if i not in eval_tasks]
h5_file_path = "/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_embeddings(embeddings, return_tensor=True):
    embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.numpy()


def plot_scatter(text_embeddings, video_embeddings, title, pca=True):
    # text embedding use +, video embedding use o
    sample_num = text_embeddings.shape[0]
    total_embeddings = np.concatenate((text_embeddings, video_embeddings), axis=0)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(total_embeddings)

    reduced_text_embeddings = reduced_embeddings[:sample_num]
    reduced_video_embeddings = reduced_embeddings[sample_num:]
    figure = plt.figure()
    plt.scatter(reduced_video_embeddings[:, 0], reduced_video_embeddings[:, 1], label="video", marker="o", zorder=1)
    plt.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], label="text", marker="+", zorder=2)
    
    plt.legend()
    if pca:
        title = title + " w PCA"
    else:
        title = title + " wo PCA"
    # I want to return the image
    plt.title(title)
    return figure


 



def main(args):
    '''
    Main function.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h5_file = h5py.File(h5_file_path, 'r')

    

    
    train_video_features = []
    train_task_name = []
    train_text_features = []
    eval_video_features = []
    eval_task_name = []
    eval_text_features = []

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    wandb_eval_task_name = "_".join([str(i) for i in eval_tasks])
    experiment_name = args.experiment_name + "_" + wandb_eval_task_name + "_seed_" + str(args.seed)
    if args.mse:
        experiment_name = experiment_name + "_mse_" + str(args.mse_weight)

    save_path = "/scr/jzhang96/roboclipv2_saved_pca_models"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.wandb:
        wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group=args.run_group,
            config=args,
            name=experiment_name,
        )

    for task in train_tasks:
        single_task = str(task)
        data_group = h5_file[single_task]
        task_name = data_group.attrs["task_name"].split("-v2")[0]
        video_idx = len(list(data_group.keys()))
        choose_idx_range = random.sample(range(video_idx - 1), args.video_sample_per_task) # xclip_text_feature also is a key name
        this_video_feature = []
        this_text_feature = []
        for idx in choose_idx_range:
            video_feature = data_group[str(idx)]["xclip_video_feature"]
            this_video_feature.append(video_feature)
            text_feature = data_group["xclip_text_feature"]
            this_text_feature.append(text_feature)
            
        this_video_feature = np.array(this_video_feature)
        this_text_feature = np.array(this_text_feature)
        train_video_features.append(this_video_feature)
        train_task_name.append(task_name)
        train_text_features.append(this_text_feature)

    train_video_features = np.concatenate(train_video_features, axis=0)
    train_text_features = np.concatenate(train_text_features, axis=0)
    train_video_features = normalize_embeddings(train_video_features)
    train_text_features = normalize_embeddings(train_text_features)


    train_img = plot_scatter(train_text_features, train_video_features, "xclip train zero-shot", pca=False)
    # save to png


    for task in eval_tasks:
        single_task = str(task)
        data_group = h5_file[single_task]
        task_name = data_group.attrs["task_name"].split("-v2")[0]
        video_idx = len(list(data_group.keys()))
        choose_idx_range = random.sample(range(video_idx - 1), args.video_sample_per_task)
        this_video_feature = []
        this_text_feature = []
        for idx in choose_idx_range:
            video_feature = data_group[str(idx)]["xclip_video_feature"]
            this_video_feature.append(video_feature)
            text_feature = data_group["xclip_text_feature"]
            this_text_feature.append(text_feature)

        this_video_feature = np.array(this_video_feature)
        this_text_feature = np.array(this_text_feature)
        eval_video_features.append(this_video_feature)
        eval_task_name.append(task_name)
        eval_text_features.append(this_text_feature)

    h5_file.close()
    eval_video_features = np.concatenate(eval_video_features, axis=0)
    eval_text_features = np.concatenate(eval_text_features, axis=0)
    eval_video_features = normalize_embeddings(eval_video_features)
    eval_text_features = normalize_embeddings(eval_text_features)

    eval_img = plot_scatter(eval_text_features, eval_video_features, "xclip eval zero-shot", pca=False)
    # save to png
    if args.wandb:
        wandb_log = {}
        wandb_log["train_img"] = wandb.Image(train_img)
        wandb_log["eval_img"] = wandb.Image(eval_img)
        wandb.log(wandb_log, step = 0)

    pca_text_model, pca_video_model = learn_pca_model(train_text_features, train_video_features)
    pca_text_model_path = os.path.join(save_path, "pca_text_model.pkl")
    pca_video_model_path = os.path.join(save_path, "pca_video_model.pkl")
    joblib.dump(pca_text_model, pca_text_model_path)
    joblib.dump(pca_video_model, pca_video_model_path)

    train_video_features = transform_embeddings(pca_video_model, train_video_features)
    train_text_features = transform_embeddings(pca_text_model, train_text_features)
    eval_video_features = transform_embeddings(pca_video_model, eval_video_features)
    eval_text_features = transform_embeddings(pca_text_model, eval_text_features)

    train_text_features = normalize_embeddings(train_text_features, return_tensor=False)
    eval_text_features = normalize_embeddings(eval_text_features, return_tensor=False)

    m_matrix = compute_M(pca_text_model, pca_video_model)
    model = get_model(normalize=args.normalize).to(device)
    model.fc.weight.data = m_matrix.T.to(device)

    train_video_trans_feature = model(torch.from_numpy(train_video_features).float().to(device))
    eval_video_trans_feature = model(torch.from_numpy(eval_video_features).float().to(device))

    train_img = plot_scatter(train_text_features, train_video_trans_feature.cpu().detach().numpy(), "xclip train zero-shot", pca=True)
    eval_img = plot_scatter(eval_text_features, eval_video_trans_feature.cpu().detach().numpy(), "xclip eval zero-shot", pca=True)

    train_img.savefig("train_xclip_zero_pca.png")
    eval_img.savefig("eval_xclip_zero_pca.png")

    loss = MILNCELoss()
    mse_loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in tqdm(range(args.epochs + 1 )):
        model.train()
        optimizer.zero_grad()
        train_transformed_video = model(torch.from_numpy(train_video_features).float().to(device))

        nce_loss = loss(train_transformed_video, torch.from_numpy(train_text_features).float().to(device))
        mse_loss = mse_loss_function(train_transformed_video, torch.from_numpy(train_text_features).float().to(device))
        if args.mse:
            train_loss = nce_loss + args.mse_weight * mse_loss
        else:
            train_loss = nce_loss
        
            
        train_loss.backward()
        optimizer.step()
        train_cosine_similarity = F.cosine_similarity(train_transformed_video, torch.from_numpy(train_text_features).float().to(device))
        wandb_log = {}

        wandb_log["train_loss"] = train_loss.item()
        wandb_log["train_cosine_similarity"] = train_cosine_similarity.mean().item()
        wandb_log["train_video_avg_norm"] = torch.norm(train_transformed_video, dim=1).mean().item()
        wandb_log["nce_loss"] = nce_loss.item()
        wandb_log["mse_loss"] = mse_loss.item()
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                train_img = plot_scatter(train_text_features, train_transformed_video.cpu().detach().numpy(), "xclip train zero-shot_{}".format(epoch), pca=True)
                wandb_log["train_img"] = wandb.Image(train_img)
                plt.close(train_img)

                train_top1, train_top3, train_top5 = compute_similarity_sorted(train_transformed_video.cpu().detach().numpy(), train_text_features, train_task_name, args.video_sample_per_task)
                wandb_log["train_video_text_top1"] = train_top1
                wandb_log["train_video_text_top3"] = train_top3
                wandb_log["train_video_text_top5"] = train_top5


        with torch.no_grad():
            model.eval()
            eval_transformed_video = model(torch.from_numpy(eval_video_features).float().to(device))

            nce_loss = loss(eval_transformed_video, torch.from_numpy(eval_text_features).float().to(device))
            mse_loss = mse_loss_function(eval_transformed_video, torch.from_numpy(eval_text_features).float().to(device))
            if args.mse:
                eval_loss = nce_loss + args.mse_weight * mse_loss
                
            else:
                eval_loss = nce_loss

            eval_cosine_similarity = F.cosine_similarity(eval_transformed_video, torch.from_numpy(eval_text_features).float().to(device))
            wandb_log["eval_cosine_similarity"] = eval_cosine_similarity.mean().item()
            wandb_log["eval_loss"] = eval_loss.item()
            wandb_log["eval_video_avg_norm"] = torch.norm(eval_transformed_video, dim=1).mean().item()
            wandb_log["eval_nce_loss"] = nce_loss.item()
            wandb_log["eval_mse_loss"] = mse_loss.item()
            
            if epoch % args.eval_freq == 0:
                eval_img = plot_scatter(eval_text_features, eval_transformed_video.cpu().detach().numpy(), "xclip eval zero-shot_{}".format(epoch), pca=True)
                wandb_log["eval_img"] = wandb.Image(eval_img)
                plt.close(eval_img)

                eval_img_with_text = plot_scatter_text(eval_transformed_video.cpu().detach().numpy(),
                                                       eval_text_features,
                                                        eval_task_name,
                                                        args.video_sample_per_task)
                wandb_log["eval_img_with_text"] = wandb.Image(eval_img_with_text)
                plt.close(eval_img_with_text)
                eval_top1, eval_top3, eval_top5 = compute_similarity_sorted(eval_transformed_video.cpu().detach().numpy(), eval_text_features, eval_task_name, args.video_sample_per_task)
                wandb_log["eval_video_text_top1"] = eval_top1
                wandb_log["eval_video_text_top3"] = eval_top3
                wandb_log["eval_video_text_top5"] = eval_top5

                mlp_model_path = os.path.join(save_path, "mlp_model_{}.pth".format(epoch))
                torch.save(model.state_dict(), mlp_model_path)

            
        if args.wandb:
            wandb.log(wandb_log, step = epoch + 1)




if __name__ == "__main__":


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_file_path', type=str, default="/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5")
    argparser.add_argument('--video_sample_per_task', type=int, default=15)
    argparser.add_argument('--epochs', type=int, default=600)
    argparser.add_argument('--lr', type=float, default=0.0001)
    argparser.add_argument('--eval_freq', type=int, default=100)
    argparser.add_argument('--run_group', type=str, default="xclip_pca_transformation_norm")
    argparser.add_argument('--experiment_name', type=str, default="xclip_pca_transformation")
    argparser.add_argument('--wandb', action="store_true")
    argparser.add_argument('--normalize', action="store_true")
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--mse', action="store_true")
    argparser.add_argument('--mse_weight', type=float, default=0.001)

    args = argparser.parse_args()
    main(args)
