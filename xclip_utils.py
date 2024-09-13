import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()


def load_model():

    model_name = "microsoft/xclip-base-patch16-zero-shot"
    xclip_tokenizer = AutoTokenizer.from_pretrained(model_name)
    xclip_net = AutoModel.from_pretrained(model_name).cuda()
    xclip_processor = AutoProcessor.from_pretrained(model_name)
    return xclip_tokenizer, xclip_net, xclip_processor

def embedding_text(list_of_text, xclip_tokenizer, xclip_net):
    text_tokens = xclip_tokenizer(list_of_text, return_tensors="pt")
    for key in text_tokens:
        text_tokens[key] = text_tokens[key].cuda()
    text_features = xclip_net.get_text_features(**text_tokens)
    return text_features




def embedding_video(video, xclip_net, xclip_processor):
    frames = xclip_processor(videos=list(video), return_tensors="pt")
    frames = frames["pixel_values"].cuda()
    video_features = xclip_net.get_video_features(frames)
    return video_features

if __name__ == "__main__":
    # embedding text
    
    xclip_tokenizer, xclip_net, xclip_processor = load_model()

    aaa = ["a", "b", "c"]
    text_embeddings = embedding_text(aaa, xclip_tokenizer, xclip_net)
    text_embeddings = normalize_embeddings(text_embeddings)
    print(text_embeddings.shape) #(3, 512)


    # embedding video
    video = np.random.randint(0, 255, (32, 224, 224, 3))
    video_embeddings = embedding_video(video, xclip_net, xclip_processor)
    video_embeddings = normalize_embeddings(video_embeddings)
    print(video_embeddings.shape) #(1, 512)
