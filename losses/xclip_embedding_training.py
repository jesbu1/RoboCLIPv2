import torch    
# from s3dg import S3D
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import argparse
import h5py
from tqdm import tqdm
import os
import imageio
from torch.utils.data import DataLoader
# from dataloader import GifDataset, GifProgressDataset, GifProgressTrainDataset
import torch.nn as nn
import torch as th
import wandb
import random
import matplotlib.pyplot as plt
import copy
from dataloader_embedding_text import GifTextEmbeddingDataset
from sklearn.decomposition import PCA
from triplet_utils import SingleLayerMLP, normalize_embeddings, triplet_loss, MILNCELoss, load_model
from plot_utils import plot_distribution, xclip_get_progress_embedding, plot_progress_embedding
from triplet_utils import compute_M
import joblib
# this file will add everything
# 1. loss type: MILNCE, Triplet, Ours, also add random noise to the embedding, random noise will add at the output of the VLM
# 2. model: xclip
# 3. augmentation

def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "triplet_loss_" + str(args.task_nums) + "_" + str(args.seed) + "_fix"



    if args.time_shuffle:
        experiment_name += "_TimeShuffle"
    if args.time_shorten:
        experiment_name += "_TimeShort"
    if args.norm_vlm:
        experiment_name += "_NormVLM"
    else:
        experiment_name += "_NoNormVLM"
    if args.random_noise:
        experiment_name += "_RandomNoise" 
    if args.rand_neg:
        experiment_name += "_RandNeg"
    if args.augmentation:
        experiment_name += "_Augmentation"
    if args.pca:
        experiment_name += "_PCA"
    
    experiment_name += args.loss_type

    h5_text_file = h5py.File(args.h5_text_path, "r")
    h5_embedding_file = h5py.File(args.h5_embedding_path, "r")
    h5_video_file = h5py.File(args.h5_video_path, "r")

    # choose button-press-topdown-v2-goal-hidden as the evaluation task
    video_group = h5_video_file["button-press-topdown-v2-goal-hidden"]
    video_names = list(video_group.keys())[-1]
    BPTD_video = np.asarray(video_group[video_names][()])
    BPTD_video = th.tensor(BPTD_video).cuda()
    BPTD_text_embedding = np.asarray(h5_text_file["button-press-topdown-v2-goal-hidden"])
    BPTD_text_embedding = np.expand_dims(BPTD_text_embedding, axis=0)
    video_group = h5_video_file["door-open-v2-goal-hidden"]
    video_names = list(video_group.keys())[-1]
    DO_video = np.asarray(video_group[video_names][()])
    DO_video = th.tensor(DO_video).cuda()
    DO_text_embedding = np.asarray(h5_text_file["door-open-v2-goal-hidden"])
    DO_text_embedding = np.expand_dims(DO_text_embedding, axis=0)
    h5_video_file.close()

    xclip_tokenizer, xclip_net, xclip_processor = load_model("xclip")
    xclip_net = xclip_net.cuda()
    xclip_net.eval()

    BPTD_embedding = xclip_get_progress_embedding(args, BPTD_video, xclip_processor, xclip_net).squeeze(1)
    DO_embedding = xclip_get_progress_embedding(args, DO_video, xclip_processor, xclip_net).squeeze(1)
    del xclip_net, xclip_processor, xclip_tokenizer


    model = SingleLayerMLP(512, 512).cuda()
    

    if args.loss_type == "MILNCE":
        loss_func = MILNCELoss()
    else:
        loss_func = triplet_loss

    if args.load_model_path is not None:
        model.load_state_dict(th.load(args.load_model_path)["model_state_dict"])
        optimizer.load_state_dict(th.load(args.load_model_path)["optimizer_state_dict"])

    evaluate_task = ["door-close-v2-goal-hidden", "door-open-v2-goal-hidden", "drawer-close-v2-goal-hidden", "button-press-v2-goal-hidden", "button-press-topdown-v2-goal-hidden"]
    evaluate_run_embeddings = []
    evaluate_run_text_embeddings = []
    total_evaluate_tasks = list(h5_embedding_file["GT_Videos"].keys())
    total_evaluate_embeddings = []

    for keys in total_evaluate_tasks:
        task_data = np.asarray(h5_embedding_file["GT_Videos"][keys])
        # random choose 15
        # choose_index = np.random.choice(task_data.shape[0], 15, replace=False)
        # task_data = task_data[:15]
        total_evaluate_embeddings.append(task_data)
    total_evaluate_embeddings = np.concatenate(total_evaluate_embeddings, axis = 0)

    for keys in evaluate_task:
        text_embedding = np.asarray(h5_text_file[keys])
        text_embedding = np.expand_dims(text_embedding, axis=0)
        task_data = np.asarray(h5_embedding_file["GT_Videos"][keys])
        # random choose 10
        choose_index = np.random.choice(task_data.shape[0], 10, replace=False)
        task_data = task_data[choose_index]
        evaluate_run_embeddings.append(task_data)
        evaluate_run_text_embeddings.append(text_embedding)
    evaluate_run_embeddings = np.concatenate(evaluate_run_embeddings, axis = 0)
    evaluate_run_text_embeddings = np.concatenate(evaluate_run_text_embeddings, axis = 0)

    total_text_embedding = []
    for keys in h5_text_file.keys():
        emb = np.asarray(h5_text_file[keys])
        emb = np.expand_dims(emb, axis=0)
        total_text_embedding.append(emb)
    total_text_embedding = np.concatenate(total_text_embedding, axis=0)
    total_text_embedding = normalize_embeddings(total_text_embedding, False)
    total_text_embedding = np.array(total_text_embedding, dtype=np.float32)


    if args.norm_vlm:
        evaluate_run_embeddings = normalize_embeddings(evaluate_run_embeddings, False)
        total_evaluate_embeddings = normalize_embeddings(total_evaluate_embeddings, False)
        BPTD_embedding = normalize_embeddings(BPTD_embedding, False)
        DO_embedding = normalize_embeddings(DO_embedding, False)
    BPTD_text_embedding = normalize_embeddings(BPTD_text_embedding, False)
    DO_text_embedding = normalize_embeddings(DO_text_embedding, False)
    evaluate_run_text_embeddings = normalize_embeddings(evaluate_run_text_embeddings, False)

    pca_text_model = None
    pca_video_model = None

    if args.pca:
        use_text_embedding = None
        for kk in range (11):
            if kk == 0:
                use_text_embedding = total_text_embedding
            else:
                use_text_embedding = np.concatenate([use_text_embedding, total_text_embedding], axis=0)
        pca_video_model = PCA(n_components=512)
        pca_text_model = PCA(n_components=512)
        pca_video_model.fit(total_evaluate_embeddings)
        pca_text_model.fit(use_text_embedding)
        # convert to numpy float type
        pca_save_path = (f"pca_loss_models/{experiment_name}")
        os.makedirs(pca_save_path, exist_ok=True)
        video_model_filename = (f"{pca_save_path}/pca_model_video.pkl")
        joblib.dump(pca_video_model, video_model_filename)

        os.makedirs(pca_save_path, exist_ok=True)
        text_model_filename = (f"{pca_save_path}/pca_model_text.pkl")
        joblib.dump(pca_text_model, text_model_filename)

        total_evaluate_embeddings = pca_video_model.transform(total_evaluate_embeddings)
        evaluate_run_embeddings = pca_video_model.transform(evaluate_run_embeddings)
        BPTD_embedding = pca_video_model.transform(BPTD_embedding)
        DO_embedding = pca_video_model.transform(DO_embedding)

        total_text_embedding = pca_text_model.transform(total_text_embedding)
        evaluate_run_text_embeddings = pca_text_model.transform(evaluate_run_text_embeddings)
        BPTD_text_embedding = pca_text_model.transform(BPTD_text_embedding)
        DO_text_embedding = pca_text_model.transform(DO_text_embedding)

        total_evaluate_embeddings = np.array(total_evaluate_embeddings, dtype=np.float32)
        evaluate_run_embeddings = np.array(evaluate_run_embeddings, dtype=np.float32)
        BPTD_embedding = np.array(BPTD_embedding, dtype=np.float32)
        DO_embedding = np.array(DO_embedding, dtype=np.float32)

        total_text_embedding = np.array(total_text_embedding, dtype=np.float32)
        evaluate_run_text_embeddings = np.array(evaluate_run_text_embeddings, dtype=np.float32)
        BPTD_text_embedding = np.array(BPTD_text_embedding, dtype=np.float32)
        DO_text_embedding = np.array(DO_text_embedding, dtype=np.float32)

        # total_evaluate_embeddings = normalize_embeddings(total_evaluate_embeddings, False)
        # evaluate_run_embeddings = normalize_embeddings(evaluate_run_embeddings, False)
        # BPTD_embedding = normalize_embeddings(BPTD_embedding, False)
        # DO_embedding = normalize_embeddings(DO_embedding, False)

        total_text_embedding = normalize_embeddings(total_text_embedding, False)
        evaluate_run_text_embeddings = normalize_embeddings(evaluate_run_text_embeddings, False)
        BPTD_text_embedding = normalize_embeddings(BPTD_text_embedding, False)
        DO_text_embedding = normalize_embeddings(DO_text_embedding, False)

    h5_text_file.close()
    h5_embedding_file.close()

    # # save pca model
    # # compute embedding distance
    # aaa = copy.deepcopy(BPTD_text_embedding)
    # bbb = copy.deepcopy(DO_text_embedding)
    # aaa = normalize_embeddings(aaa, False)
    # bbb = normalize_embeddings(bbb, False)
    # dis = np.linalg.norm(aaa - bbb, axis=1)
    # # aaa norm 
    # aaa = np.linalg.norm(aaa, axis=1)


    # import pdb ; pdb.set_trace()


    # init MLP model
    if args.pca:
        computed_matrix = compute_M(pca_video_model.components_, pca_text_model.components_)
        with th.no_grad():
            model.linear.weight = nn.Parameter(computed_matrix.T.cuda().float())
            model.linear.bias = nn.Parameter(th.zeros(512).cuda())

    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="debug_text_embedding_triplet_xclip",
        config=args,
        name=experiment_name,
    )

    dataset = GifTextEmbeddingDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for idx, data in enumerate(tqdm(dataloader)):
            gt_features = data["gt_array"].cuda().squeeze().float() # text
            pos_features = data["pos_array"].cuda().squeeze().float()
            neg_features = data["neg_array"].cuda().squeeze().float()
            progress = data["progress"].cuda().float()
            neg_type = data["type"]
            gt_features = normalize_embeddings(gt_features).float()
            if args.norm_vlm:
                pos_features = normalize_embeddings(pos_features).float()
                neg_features = normalize_embeddings(neg_features).float() # already type 4 loss already add random noise

            if args.pca:
                gt_features = pca_text_model.transform(gt_features.cpu().detach().numpy())
                pos_features = pca_video_model.transform(pos_features.cpu().detach().numpy())
                neg_features = pca_video_model.transform(neg_features.cpu().detach().numpy())

                
                # pos_features = normalize_embeddings(pos_features, False)
                # neg_features = normalize_embeddings(neg_features, False)
            gt_features = normalize_embeddings(gt_features, False)
            gt_features = th.tensor(gt_features).cuda().float()
            pos_features = th.tensor(pos_features).cuda().float()
            neg_features = th.tensor(neg_features).cuda().float()

            pos_features = model(pos_features)
            neg_features = model(neg_features)

            # find hardest negative
            if not args.rand_neg:
                # choose the hardest negative
                type_1_mask = (neg_type == 1)
                type_1_gt_features = gt_features[type_1_mask]
                cosine_similarity_matrix = torch.mm(type_1_gt_features, pos_features.t())
                true_indices = np.where(type_1_mask)[0]
                for i, true_index in enumerate(true_indices):
                    cosine_similarity_matrix[i, true_index] = -2.0
                hardest_neg_indices = torch.argmax(cosine_similarity_matrix, dim=1)
                neg_features[type_1_mask] = pos_features[hardest_neg_indices].clone()

            # gt, positive, negative, type, margin = (1.0, 1.0, 1.0, 0.0, 1.0), progress = None
            if args.loss_type == "triplet":
                loss = loss_func(                
                    gt = gt_features, 
                    positive = pos_features, 
                    negative = neg_features, 
                    type = neg_type, 
                    margin = (1.5, 1.0, 1.0, 0.0, 1.0), # hardest 1.5, shuffle 1.0, shorten range from 1.0 to 0.0, random noise 1.0
                    progress = progress)
            elif args.loss_type == "MILNCE":
                loss = loss_func(gt_features, pos_features)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log average gradient
            # total_grad = 0
            # for name, param in model.named_parameters():
            #     total_grad += param.grad.abs().mean().item()
            # wandb.log({"grad/average_grad": total_grad / len(list(model.parameters()))})
            #compute norm
            # aaa = gt_features[0:1]
            # aaa_norm = torch.norm(aaa, p=2, dim=1).mean().item()

            # import pdb ; pdb.set_trace()
            with torch.no_grad():
                pos_cos_avg = F.cosine_similarity(gt_features, pos_features, dim=1).mean().item()
                neg_cos_avg = F.cosine_similarity(gt_features, neg_features, dim=1).mean().item()

                pos_l2_dis = torch.norm(gt_features - pos_features, p=2, dim=1).mean().item()
                neg_l2_dis = torch.norm(gt_features - neg_features, p=2, dim=1).mean().item()    

                wandb.log({
                    "train_loss": loss.item(),
                    "pos_cos_avg": pos_cos_avg,
                    "neg_cos_avg": neg_cos_avg,
                    "pos_l2_dis": pos_l2_dis,
                    "neg_l2_dis": neg_l2_dis,
                })

        if epoch % 100 == 99:
        # if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                # plot distribution
                figure_1, figure_2 = plot_distribution(
                    model, 
                    evaluate_run_embeddings, 
                    total_evaluate_embeddings, 
                    evaluate_task,
                    total_evaluate_tasks, 
                    evaluate_run_text_embeddings,
                )

                BPTD_progress_figure = plot_progress_embedding(
                    BPTD_embedding, 
                    model, 
                    BPTD_text_embedding)

                DO_progress_figure = plot_progress_embedding(
                    DO_embedding, 
                    model, 
                    DO_text_embedding)
                
                wandb.log({
                    "total_dist": wandb.Image(figure_1),
                    "eval_task_dist": wandb.Image(figure_2),
                    "progress/BPTD_progress_figure": wandb.Image(BPTD_progress_figure),
                    "progress/DO_progress_figure": wandb.Image(DO_progress_figure),
                })
                plt.close(figure_1)
                plt.close(figure_2)
                plt.close(BPTD_progress_figure)
                plt.close(DO_progress_figure)

            save_path = os.path.join("/scr/jzhang96/triplet_text_loss_models", experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            th.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(save_path, "model_" + str(epoch) + ".pth"))

            

    run.finish()







if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--h5_embedding_path', type=str, default='/scr/jzhang96/metaworld_xclip_all_embedding_15.h5')
    argparser.add_argument('--h5_text_path', type=str, default='metaworld_xclip_text.h5')
    argparser.add_argument('--h5_video_path', type=str, default='/scr/jzhang96/metaworld_gifs_1.h5')
    argparser.add_argument('--time_shuffle', action='store_true')
    argparser.add_argument('--time_shorten', action='store_true')
    argparser.add_argument('--random_noise', action='store_true')
    # argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--rand_neg', action='store_true')
    argparser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'MILNCE'])
    argparser.add_argument('--task_nums', type=int, default=50)
    argparser.add_argument('--augmentation', action='store_true')
    argparser.add_argument('--norm_vlm', action='store_true')
    argparser.add_argument('--load_model_path', type=str, default=None)
    argparser.add_argument('--start_epoch', type=int, default=0)
    argparser.add_argument('--pca', action='store_true')

    args = argparser.parse_args()
    main(args)
