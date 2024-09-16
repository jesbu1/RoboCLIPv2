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
from mrr_utils import get_xclip_embeddings, reduce_dimension, compute_M

# model_name = "microsoft/xclip-base-patch16-zero-shot"
# xclip_tokenizer = AutoTokenizer.from_pretrained(model_name)
# xclip_net = AutoModel.from_pretrained(model_name).cuda()
# xclip_processor = AutoProcessor.from_pretrained(model_name)
# xclip_net.eval()


class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)

def triplet_loss(gt, positive, negative, type, margin = (1.0, 1.0, 1.0, 0.0), progress = None): 

    pos_sim = cosine_similarity(gt, positive)
    neg_sim = cosine_similarity(gt, negative)

    loss = torch.zeros_like(pos_sim).cuda()

    # type1: hard negative margin = 1.5
    mask_type_1 = (type == 1)
    loss[mask_type_1] = F.relu(margin[0] - pos_sim[mask_type_1] + neg_sim[mask_type_1])

    # type2: semi-hard negative margin = 1.2
    mask_type_2 = (type == 2)
    loss[mask_type_2] = F.relu(margin[1] - pos_sim[mask_type_2] + neg_sim[mask_type_2])

    # type3: adaptive margin
    mask_type_3 = (type == 3)
    progress = progress[mask_type_3]
    # adaptive margin range from 1.0 to 0.0
    adaptive_margin = (margin[2] + (margin[3] - margin[2]) * progress).cuda()
    loss[mask_type_3] = F.relu(adaptive_margin - pos_sim[mask_type_3] + neg_sim[mask_type_3])

    return loss.mean()


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().cpu().numpy()


def plot_progress(array, s3d_model, transform_model):

    array_length = array.shape[0]
    similarities = []
    with th.no_grad():
        s3d_model.eval()
        transform_model.eval()

        copy_array = copy.deepcopy(array)
        indices = np.linspace(0, len(copy_array) - 1, 32).astype(int)
        copy_array = copy_array[indices]
        copy_array = th.tensor(copy_array).cuda()
        copy_array = copy_array.float().permute(3, 0, 1, 2).unsqueeze(0)
        copy_array = copy_array / 255.0
        copy_array = s3d_model(copy_array)["video_embedding"]
        GT_embedding = normalize_embeddings(copy_array)
        GT_embedding = transform_model(GT_embedding)

        for i in range(32, array_length + 1):
            copy_array = copy.deepcopy(array)
            copy_array = copy_array[:i]
            progress = i / array_length
            indices = np.linspace(0, i - 1, 32).astype(int)
            copy_array = copy_array[indices]
            copy_array = th.tensor(copy_array).cuda()
            copy_array = copy_array.float().permute(3, 0, 1, 2).unsqueeze(0)
            copy_array = copy_array / 255.0
            copy_array = s3d_model(copy_array)["video_embedding"]
            progress_embedding = normalize_embeddings(copy_array)
            progress_embedding = transform_model(progress_embedding)
            similarity = cosine_similarity(GT_embedding, progress_embedding)

            similarities.append((i, similarity.item()))

    transform_model.train()
    figure_1 = plt.figure()
    plt.plot([i[0] for i in similarities], [i[1] for i in similarities])
    plt.xlabel("Length")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity of GT embedding with progress embedding")

    return figure_1



def plot_progress_xclip(array, xclip_processor, xclip_net, transform_model, text_array, pca_text, pca_video):
    # text_array already tensor
    text_array = normalize_embeddings(text_array.clone())
    text_array = pca_text.transform(text_array.cpu())
    text_array = normalize_embeddings(text_array).float().cuda()

    array_length = array.shape[0]
    similarities = []
    with th.no_grad():
        # s3d_model.eval()
        transform_model.eval()


        for i in range(32, array_length + 1):
            copy_array = copy.deepcopy(array)
            copy_array = copy_array[:i]
            progress = i / array_length
            indices = np.linspace(0, i - 1, 32).astype(int)
            copy_array = copy_array[indices]
            copy_array = copy_array[:, 13:237, 13:237, :]
            copy_array = xclip_processor(videos = list(copy_array), return_tensors="pt").pixel_values.cuda()
            copy_array = xclip_net.get_video_features(copy_array)
            copy_array = normalize_embeddings(copy_array)
            copy_array = pca_video.transform(copy_array.cpu())
            copy_array = normalize_embeddings(copy_array).float().cuda()

            copy_array = transform_model(copy_array)
            similarity = cosine_similarity(text_array, copy_array)

            similarities.append((i, similarity.item()))

    transform_model.train()
    figure_1 = plt.figure()
    plt.plot([i[0] for i in similarities], [i[1] for i in similarities])
    plt.xlabel("Length")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity of GT embedding with progress embedding")

    return figure_1





def plot_distribution(transform_model, evaluate_run_embeddings, total_evaluate_embeddings, evaluate_tasks, total_evaluate_tasks, eval_text_embedding, pca_text, pca_video):




    eval_text_embedding = pca_text.transform(eval_text_embedding.clone())
    evaluate_run_embeddings = pca_video.transform(evaluate_run_embeddings.cpu().clone())
    total_evaluate_embeddings = pca_video.transform(total_evaluate_embeddings.cpu().clone())
    evaluate_run_embeddings = normalize_embeddings(evaluate_run_embeddings, True)
    total_evaluate_embeddings = normalize_embeddings(total_evaluate_embeddings, True)


    total_evaluate_embeddings = transform_model(total_evaluate_embeddings.cuda().float()).detach().cpu().numpy()
    evaluate_run_embeddings = transform_model(evaluate_run_embeddings.cuda().float()).detach().cpu().numpy()
    eval_text_embedding = normalize_embeddings(eval_text_embedding, False)
    evaluate_run_embeddings = normalize_embeddings(evaluate_run_embeddings, False)
    total_evaluate_embeddings = normalize_embeddings(total_evaluate_embeddings, False)


    total_embedding = np.concatenate([total_evaluate_embeddings, eval_text_embedding], axis=0)
    pca = PCA(n_components=2)
    pca_model = pca.fit(total_embedding)
    total_video_embedding = pca_model.transform(total_evaluate_embeddings)
    run_video_embedding = pca_model.transform(evaluate_run_embeddings)
    eval_text_embedding = pca_model.transform(eval_text_embedding)

    figure_1 = plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(total_evaluate_tasks)))  
    for i in range(len(total_evaluate_tasks)):
        group_data = total_video_embedding[i*10:(i+1)*10]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = total_evaluate_tasks[i]
        plt.scatter(x, y, color=colors[i], label=text_name)
    
    plt.title('2D PCA for Metaworld Total Videos')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')
    # plt.legend(loc='upper left', ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 1]) # adjust the plot to the right (to fit the legend)
    

    figure_2 = plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(evaluate_tasks)))
    for i in range(len(evaluate_tasks)):
        group_data = run_video_embedding[i*10:(i+1)*10]
        x = group_data[:, 0]
        y = group_data[:, 1]
        text_name = evaluate_tasks[i].split("-v2")[0]
        text_embedding = eval_text_embedding[i]
        plt.scatter(x, y, color=colors[i], label=text_name, marker='o', s=100, zorder=2)
        # put "x" above the point
        plt.scatter(text_embedding[0], text_embedding[1], color=colors[i], marker='x', s=100, zorder=3)

    plt.title('2D PCA for Metaworld Evaluate Videos')
    plt.xlabel('x-dim')
    plt.ylabel('y-dim')
    plt.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    plt.tight_layout() # adjust the plot to the right (to fit the legend)

    return figure_1, figure_2


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)


    
def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    eval_tasks = []

    h5_file = h5py.File(args.h5_path, "r")
    model_name = args.model_name

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    # training_num = str(50 - len(eval_tasks))
    training_num = str(10)
    experiment_name = "triplet_loss_" + "PCA_" + training_num + "_" + str(args.seed) + "_" + args.model_name

    if args.time_shuffle:
        experiment_name += "_TimeShuffle"
    if args.time_shorten:
        experiment_name += "_TimeShort"
    if args.norm:
        experiment_name += "_Norm"

    if args.random_noise:
        experiment_name += "_RandomNoise" 
    if args.rand_neg:
        experiment_name += "_RandNeg"
    experiment_name += args.loss_type
    

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
        group="text_adaptive_triplet_xclip_rerun_0.9",
        config=args,
        name=experiment_name,
    )

    if model_name == "xclip":
        xclip_net = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
        # xclip_net = torch.compile(xclip_net)
        xclip_net.eval().cuda()
        # pixel_values = self.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)
        xclip_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
        # h5_xclip_embedding_file = h5py.File("/scr/jzhang96/metaworld_25_generated_xclip_embeddings.h5", "r")
        # train_xclip_video, train_xclip_text, train_xclip_mappings = get_xclip_embeddings(task_id=set(range(50)))
        train_xclip_video, train_xclip_text, train_xclip_mappings = get_xclip_embeddings(task_id=set(13,))

        pca_text, reduced_train_text = reduce_dimension(train_xclip_text.cpu(), variance_threshold=args.variance_threshold,
                                                        embed_type='text', seed=args.seed, kernel='linear',
                                                        val_task_name='Fully_supervised', exp_name = experiment_name)  # pca_emb=pca_train_alltext
        pca_video, reduced_train_video = reduce_dimension(train_xclip_video.cpu(), variance_threshold=args.variance_threshold,
                                                          embed_type='video', dimension=reduced_train_text.shape[1],
                                                          seed=args.seed, kernel='linear',
                                                          val_task_name='Fully_supervised', exp_name = experiment_name)  # 35，512
        computed_matrix = compute_M(pca_video.components_, pca_text.components_,
                                    variance_threshold=args.variance_threshold, seed=args.seed)

    else:
        s3d_model = S3D('../s3d_dict.npy', 512)
        s3d_model.load_state_dict(th.load('../s3d_howto100m.pth'))
        s3d_model.eval().cuda()

    transform_model = SingleLayerMLP(reduced_train_text.shape[1], reduced_train_video.shape[1], normalize=True).cuda()
    with th.no_grad():
        transform_model.linear.weight = nn.Parameter(computed_matrix.T.cuda().float())
        transform_model.linear.bias = nn.Parameter(th.zeros(reduced_train_text.shape[1]).cuda())
    dataset = GifTextDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # loss_func = nn.TripletMarginLoss(margin=0.5, p=1)
    optimizer = th.optim.Adam(transform_model.parameters(), lr=1e-4)
    # optimizer = th.optim.Adam(transform_model.parameters(), lr=1e-3)

    if args.loss_type == "MILNCE":
        loss_func = MILNCELoss()

    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            gt_features = batch["gt_array"]
            pos_array = batch["pos_array"].cuda()
            neg_array = batch["neg_array"].cuda()
            batch_size = neg_array.shape[0]
            gt_features = normalize_embeddings(gt_features)
            gt_features = torch.from_numpy(pca_text.transform(gt_features)).cuda().float()
            gt_features = normalize_embeddings(gt_features)
            samples = torch.cat([pos_array, neg_array]).cuda()
            types = batch["type"]
            progress = batch["progress"].float().cuda()
            
            with th.no_grad():
                if model_name == "xclip":
                    # pixel_values = xclip_processor.processor(videos = list(array), return_tensors="pt").pixel_values.squeeze(0)
                    video_features = xclip_net.get_video_features(samples)

                else:
                    video_features = s3d_model(samples)["video_embedding"]

            video_features = normalize_embeddings(video_features)
            video_features = torch.from_numpy(pca_video.transform(video_features.cpu())).cuda().float()
            pos_org_feature = video_features[:batch_size].clone()
            neg_org_feature = video_features[batch_size:2*batch_size].clone()

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




        transform_model.eval()
        with torch.no_grad():
            total_figure, single_figure = plot_distribution(transform_model, 
                                                            evaluate_run_embeddings, 
                                                            total_evaluate_embeddings, 
                                                            evaluate_task, 
                                                            total_evaluate_tasks,
                                                            eval_text_embedding,
                                                            pca_text,
                                                            pca_video
                                                            )
            wandb.log({"total_total_distribution": wandb.Image(total_figure)})
            wandb.log({"eval_task_distribution": wandb.Image(single_figure)})
            plt.close(total_figure)
            plt.close(single_figure)

        transform_model.train()

        if epoch % 5 == 0:
            if args.model_name == "xclip":
                if not os.path.exists(f"/scr/jzhang96/triplet_text_loss_models/{experiment_name}"):
                    os.makedirs(f"/scr/jzhang96/triplet_text_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(), f"/scr/jzhang96/triplet_text_loss_models/{experiment_name}/{epoch}.pth")
            else:
                if not os.path.exists(f"/scr/jzhang96/triplet_text_loss_models/{experiment_name}"):
                    os.makedirs(f"/scr/jzhang96/triplet_text_loss_models/{experiment_name}")
                th.save(transform_model.state_dict(), f"/scr/jzhang96/triplet_text_loss_models/{experiment_name}/{epoch}.pth")


            seen_plt = plot_progress_xclip(seen_array, xclip_processor, xclip_net, transform_model, seen_text_embedding, pca_text, pca_video)
            unseen_plt = plot_progress_xclip(unseen_array, xclip_processor, xclip_net, transform_model, unseen_text_embedding, pca_text, pca_video)
            wandb.log({"progress/seen": wandb.Image(seen_plt)})
            wandb.log({"progress/unseen": wandb.Image(unseen_plt)})
            plt.close(seen_plt)
            plt.close(unseen_plt)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='s3d', choices=['xclip', 's3d'])
    argparser.add_argument('--time_shuffle', action='store_true')
    argparser.add_argument('--h5_path', type=str, default='/scr/jzhang96/metaworld_gifs_1.h5')
    argparser.add_argument('--time_shorten', action='store_true')
    argparser.add_argument('--norm', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--add_lower_bound', action='store_true')
    argparser.add_argument('--random_noise', action='store_true')
    argparser.add_argument('--progress_area', type=float, default=0)
    argparser.add_argument('--variance_threshold', type=float, default=512)
    argparser.add_argument('--rand_neg', action='store_true')
    argparser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'MILNCE'])
    args = argparser.parse_args()
    main(args)