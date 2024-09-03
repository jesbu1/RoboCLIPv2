import torch as th
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import imageio
import wandb
import io
import os
from sklearn.decomposition import PCA
from triplet_utils import cosine_similarity


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



def plot_progress_xclip(array, xclip_processor, xclip_net, transform_model, text_array, norm_vlm=False):
    # text_array already tensor
    text_array = normalize_embeddings(text_array)
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
            if norm_vlm:
                copy_array = normalize_embeddings(copy_array)
            # copy_array = normalize_embeddings(copy_array)
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

def xclip_get_progress_embedding(args, array, xclip_processor, xclip_net):
    array_length = array.shape[0]
    embeddings = []
    with th.no_grad():
        for i in range(32, array_length + 1):
            copy_array = copy.deepcopy(array)
            copy_array = copy_array[:i]
            indices = np.linspace(0, i - 1, 32).astype(int)
            copy_array = copy_array[indices]
            copy_array = copy_array[:, 13:237, 13:237, :]
            copy_array = xclip_processor(videos = list(copy_array), return_tensors="pt").pixel_values.cuda()
            copy_array = xclip_net.get_video_features(copy_array)
            embeddings.append(copy_array)
        embeddings = th.stack(embeddings).squeeze()
        if args.norm_vlm:
            embeddings = normalize_embeddings(embeddings)

    return embeddings

    
def plot_progress_embedding(embedding, transform_model, text_embedding):
    # text_array already tensor
    text_embedding = normalize_embeddings(text_embedding).clone().float().cuda()
    embedding_length = embedding.shape[0]
    text_array = text_embedding.repeat(embedding_length, 1).cuda()
    
    # similarities = []
    # if embedding type is not tensor convert to tensor
    if not isinstance(embedding, th.Tensor):
        embedding = th.tensor(embedding).cuda().float()

    use_embedding = embedding.clone()
    transform_model.eval()
    with th.no_grad():
        # s3d_model.eval()
        
        use_embedding = transform_model(use_embedding)
        similarity = cosine_similarity(text_array, use_embedding)
        similarities = [(i + 32, similarity[i].item()) for i in range(embedding_length)]
    transform_model.train()
    figure_1 = plt.figure()
    plt.plot([i[0] for i in similarities], [i[1] for i in similarities])
    plt.xlabel("Length")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity of GT embedding with progress embedding")
    plt.tight_layout()
    del text_array
    del use_embedding
    return figure_1






def plot_distribution(transform_model, evaluate_run_embeddings, total_evaluate_embeddings, evaluate_tasks, total_evaluate_tasks, eval_text_embedding):

    total_evaluate_embeddings = transform_model(th.tensor(total_evaluate_embeddings.copy()).cuda().float()).detach().cpu().numpy()
    evaluate_run_embeddings = transform_model(th.tensor(evaluate_run_embeddings.copy()).cuda().float()).detach().cpu().numpy()
    eval_text_embedding = normalize_embeddings(eval_text_embedding, False)
    # evaluate_run_embeddings = normalize_embeddings(evaluate_run_embeddings, False)
    # total_evaluate_embeddings = normalize_embeddings(total_evaluate_embeddings, False)


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


def save_tensor_as_gif(tensor, filename):
    # Ensure the tensor is on CPU and convert it to a NumPy array
    tensor = tensor.cpu().detach().numpy()
    
    # Check the tensor shape
    if tensor.shape[1] != 3:
        raise ValueError("Tensor shape should be (32, 3, 224, 224)")

    # Reshape from (32, 3, 224, 224) to (32, 224, 224, 3)
    tensor = np.transpose(tensor, (0, 2, 3, 1))

    # Normalize to 0-255 if necessary
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)

    # Save as GIF
    imageio.mimsave(filename, tensor, format='gif')


def log_tensor_as_gif(tensor, name):
    # Ensure the tensor is on CPU and convert it to a NumPy array
    tensor = tensor.cpu().detach().numpy()

    # Check the tensor shape
    if tensor.shape[1] != 3:
        raise ValueError("Tensor shape should be (32, 3, 224, 224)")

    # Reshape from (32, 3, 224, 224) to (32, 224, 224, 3)
    tensor = np.transpose(tensor, (0, 2, 3, 1))

    # Normalize to 0-255 if necessary
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)

    # Save to an in-memory buffer
    buffer = io.BytesIO()
    imageio.mimsave(buffer, tensor, format='gif')
    buffer.seek(0)

    # Log GIF to wandb
    wandb.log({name: wandb.Video(buffer, fps=10, format="gif")})


