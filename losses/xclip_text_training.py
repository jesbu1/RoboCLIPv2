import torch    
from s3dg import S3D
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
from dataloader_text import GifTextDataset
from sklearn.decomposition import PCA
from triplet_utils import AugmentationPipeline, SingleLayerMLP, normalize_embeddings, triplet_loss, MILNCELoss
from plot_utils import plot_distribution, plot_progress_xclip, save_tensor_as_gif, log_tensor_as_gif



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

    eval_tasks = []

    model_name = args.model_name

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    # training_num = str(50 - len(eval_tasks))
    experiment_name = "triplet_loss_" + str(args.task_nums) + "_" + str(args.seed) + "_" + args.model_name

    if args.time_shuffle:
        experiment_name += "_TimeShuffle"
    if args.time_shorten:
        experiment_name += "_TimeShort"
    if args.norm:
        experiment_name += "_Norm"
    if args.norm_vlm:
        experiment_name += "_NormVLM"
    else:
        experiment_name += "_NoNormVLM"

    if args.random_noise:
        experiment_name += "_RandomNoise" 
    if args.rand_neg:
        experiment_name += "_RandNeg"
    experiment_name += args.loss_type

    if args.augmentation:
        # augmentation_method = AugmentationPipeline(device = "cuda", strength='normal')
        augmentation_method = AugmentationPipeline(device = "cuda", strength='weak')
        experiment_name += "_Aug"
    


    

    h5_dataset_file = h5py.File(args.h5_path, "r")

    if args.model_name == "s3d":
        evaluate_h5 = h5py.File("metaworld_s3d_embedding.h5", "r")
    else:
        evaluate_h5 = h5py.File("metaworld_xclip_video_embedding.h5", "r")

    evaluate_task = ["door-close-v2-goal-hidden", "door-open-v2-goal-hidden", "drawer-close-v2-goal-hidden", "button-press-v2-goal-hidden", "button-press-topdown-v2-goal-hidden"]
    total_evaluate_tasks = list(evaluate_h5.keys())
    total_evaluate_embeddings = []
    evaluate_run_embeddings = []
    for keys in evaluate_h5.keys():
        task_data = np.asarray(evaluate_h5[keys])
        # random choose 10
        choose_index = np.random.choice(task_data.shape[0], 10, replace=False)
        task_data = task_data[choose_index]
        total_evaluate_embeddings.append(task_data)
    total_evaluate_embeddings = np.concatenate(total_evaluate_embeddings, axis = 0)



    for keys in evaluate_task:
        task_data = np.asarray(evaluate_h5[keys])
        # random choose 10
        choose_index = np.random.choice(task_data.shape[0], 10, replace=False)
        task_data = task_data[choose_index]
        evaluate_run_embeddings.append(task_data)
    evaluate_run_embeddings = np.concatenate(evaluate_run_embeddings, axis = 0)

    total_evaluate_embeddings = torch.tensor(total_evaluate_embeddings).cuda()
    evaluate_run_embeddings = torch.tensor(evaluate_run_embeddings).cuda()
    evaluate_h5.close()
    if args.model_name == "s3d":
        text_h5 = h5py.File("/scr/jzhang96/metaworld_s3d_text.h5", "r")
    else:
        text_h5 = h5py.File("metaworld_xclip_text.h5", "r")


    key = "door-close-v2-goal-hidden"
    unseen_array = np.array(h5_dataset_file[key]['output_gif_18.gif'])
    unseen_text_embedding = np.asarray(text_h5[key])
    unseen_text_embedding = torch.tensor(unseen_text_embedding).cuda()
    unseen_text_embedding = unseen_text_embedding.unsqueeze(0)


    key = "door-open-v2-goal-hidden"
    seen_array = np.array(h5_dataset_file[key]['output_gif_18.gif'])
    seen_text_embedding = np.asarray(text_h5[key])
    seen_text_embedding = torch.tensor(seen_text_embedding).cuda()
    seen_text_embedding = seen_text_embedding.unsqueeze(0)



    eval_text_embedding = []
    for keys in evaluate_task:
        embedding = np.asarray(text_h5[keys])
        embedding = np.expand_dims(embedding, axis=0)
        eval_text_embedding.append(embedding)
    eval_text_embedding = np.concatenate(eval_text_embedding, axis=0)
    eval_text_embedding = normalize_embeddings(eval_text_embedding)
    
    text_h5.close()

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="aug_text_adaptive_triplet_xclip",
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
    transform_model = SingleLayerMLP(512, 512, normalize=True).cuda()
    dataset = GifTextDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # loss_func = nn.TripletMarginLoss(margin=0.5, p=1)
    optimizer = th.optim.Adam(transform_model.parameters(), lr=2e-4)
    # optimizer = th.optim.Adam(transform_model.parameters(), lr=1e-3)

    if args.loss_type == "MILNCE":
        loss_func = MILNCELoss()

    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            gt_features = batch["gt_array"].cuda()
            pos_array = batch["pos_array"].cuda()
            neg_array = batch["neg_array"].cuda()
            batch_size = neg_array.shape[0]
            gt_features = normalize_embeddings(gt_features)
            samples = torch.cat([pos_array, neg_array]).cuda()
            types = batch["type"]
            progress = batch["progress"].float().cuda()

            augmentation = False
            if args.augmentation:
                random_value = np.random.rand()
                threshold = 1/3  # 33.3% chance (1 out of 3)
                if random_value < threshold:
                    augmentation = False  
                else:
                    augmentation = True  

            if augmentation:
                samples = augmentation_method(samples)




            # video1 = samples[0]
            # video2 = samples[1]
            # log_tensor_as_gif(video1, f"video1_{i}")
            # log_tensor_as_gif(video2, f"video2_{i}")
            # import pdb ; pdb.set_trace()



            with th.no_grad():
                if model_name == "xclip":
                    # pixel_values = xclip_processor.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)
                    video_features = xclip_net.get_video_features(samples)

                else:
                    video_features = s3d_model(samples)["video_embedding"]
            if args.norm_vlm:
                video_features = normalize_embeddings(video_features)


            video_features = transform_model(video_features) # normalization already done in the model
            pos_features = video_features[:batch_size]
            neg_features = video_features[batch_size:2*batch_size]



            if args.loss_type == "triplet":
                if not args.rand_neg:
                    # choose the hardest negative
                    type_1_mask = (types == 1)
                    type_1_gt_features = gt_features[type_1_mask]
                    cosine_similarity_matrix = torch.mm(type_1_gt_features, pos_features.t())
                    true_indices = np.where(type_1_mask)[0]
                    for i, true_index in enumerate(true_indices):
                        cosine_similarity_matrix[i, true_index] = -1.0
                    hardest_neg_indices = torch.argmax(cosine_similarity_matrix, dim=1)
                    neg_features[type_1_mask] = pos_features[hardest_neg_indices].clone()

                    if args.random_noise:
                        type_4_mask = (types == 4)
                        type_4_pos_features = pos_features[type_4_mask].clone()
                        std = 0.2
                        noise = std * torch.randn_like(type_4_pos_features)
                        noisy_type_4_pos_features = type_4_pos_features + noise
                        noisy_type_4_pos_features = normalize_embeddings(noisy_type_4_pos_features)
                        neg_features[type_4_mask] = noisy_type_4_pos_features
                    
                loss = triplet_loss(gt_features, pos_features, neg_features, types, progress=progress)
            
            else:
                loss = loss_func(gt_features, pos_features)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pos_cos_avg = F.cosine_similarity(gt_features, pos_features, dim=1).mean().item()
                neg_cos_avg = F.cosine_similarity(gt_features, neg_features, dim=1).mean().item()

                pos_l2_dis = torch.norm(gt_features - pos_features, p=2, dim=1).mean().item()
                neg_l2_dis = torch.norm(gt_features - neg_features, p=2, dim=1).mean().item()

            wandb_log = {"loss": loss.item(),
                         "cos/pos_cos_avg": pos_cos_avg,
                         "cos/neg_cos_avg": neg_cos_avg,
                         "l2dis/pos_l2_dis": pos_l2_dis,
                         "l2dis/neg_l2_dis": neg_l2_dis,
                        }

            wandb.log(wandb_log)


        if epoch % 10 == 0:
            if args.model_name == "xclip":
                if not os.path.exists(f"/home/jzhang96/triplet_text_loss_models/{experiment_name}"):
                    os.makedirs(f"/home/jzhang96/triplet_text_loss_models/{experiment_name}")
                th.save(
                    {'model_state_dict': transform_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()},
                    f"/home/jzhang96/triplet_text_loss_models/{experiment_name}/{epoch}.pth")
            else:
                if not os.path.exists(f"/home/jzhang96/triplet_text_loss_models/{experiment_name}"):
                    os.makedirs(f"/home/jzhang96/triplet_text_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(), f"/home/jzhang96/triplet_text_loss_models/{experiment_name}/{epoch}.pth")

            transform_model.eval()
            with torch.no_grad():
                total_figure, single_figure = plot_distribution(transform_model, 
                                                                evaluate_run_embeddings, 
                                                                total_evaluate_embeddings, 
                                                                evaluate_task, 
                                                                total_evaluate_tasks,
                                                                eval_text_embedding,
                                                                )
                wandb.log({"total_total_distribution": wandb.Image(total_figure)})
                wandb.log({"eval_task_distribution": wandb.Image(single_figure)})
                plt.close(total_figure)
                plt.close(single_figure)

                seen_plt = plot_progress_xclip(seen_array, xclip_processor, xclip_net, transform_model, seen_text_embedding)
                unseen_plt = plot_progress_xclip(unseen_array, xclip_processor, xclip_net, transform_model, unseen_text_embedding)
            wandb.log({"progress/seen": wandb.Image(seen_plt)})
            wandb.log({"progress/unseen": wandb.Image(unseen_plt)})
            plt.close(seen_plt)
            plt.close(unseen_plt)

            transform_model.train()



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='s3d', choices=['xclip', 's3d'])
    argparser.add_argument('--time_shuffle', action='store_true')
    argparser.add_argument('--h5_path', type=str, default='/scr/jzhang96/metaworld_gifs_1.h5')
    argparser.add_argument('--time_shorten', action='store_true')
    argparser.add_argument('--random_noise', action='store_true')
    argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--rand_neg', action='store_true')
    argparser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'MILNCE'])
    argparser.add_argument('--task_nums', type=int, default=50)
    argparser.add_argument('--augmentation', action='store_true')
    argparser.add_argument('--norm_vlm', action='store_true')

    args = argparser.parse_args()
    main(args)