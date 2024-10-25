from liv import load_liv
import torch
import json
import h5py
from clip_utils import load_model, normalize_embeddings, compute_similarity

import matplotlib.pyplot as plt
import wandb
import imageio
import numpy as np
from PIL import Image
from animation_utils import animate_video_with_rewards, log_gif_to_wandb
import clip
import torchvision.transforms as T
import matplotlib
from PIL import Image
matplotlib.use('Agg')



def embedding_image(model, processor, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(image) != Image.Image:
        image = Image.fromarray(image.astype(np.uint8))
    if processor is not None:
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    else:
        image = T.ToTensor()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embeddings = model(input=image, modality="vision")
    return image_embeddings

def embedding_text(model, tokenizer, text):
    if type(text) != list:
        text = [text]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is not None:
        inputs = tokenizer(text=text, return_tensors="pt", padding=True)
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeddings = model.get_text_features(**inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    else:
        text = clip.tokenize(text).to(device)
        text_embeddings = model(input=text, modality="text")
    return text_embeddings


def plot_videos(model_name, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, processor, tokenizer = load_model(model_name)
    video_base_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos"
    video_idxs = ["1", "2"]
    diffs = ["all_fail", "close_succ", "success", "GT"]
    tasks = ["button_press_wall", "topdown", "windowclose"]

    texts = {"button_press_wall": "Robot pressing button from side",
             "topdown": "Robot pressing button from top",
             "windowclose": "Robot closing window"}

    # texts = {"button_press_wall": "Pushing the button from the side",
    #          "topdown": "Pressing button from top",
    #          "windowclose": "Closing window"}


    for task in tasks:
        text = texts[task]
        text_embeddings = embedding_text(model, tokenizer, text).to(device).float()
        text_embeddings = normalize_embeddings(text_embeddings)


        for diff in diffs:
            for video_idx in video_idxs:
                gif_path = f"{video_base_path}/{task}/{diff}/{video_idx}.gif"
                # load gif
                frames = imageio.mimread(gif_path)
                frames = [frame[:,:,0:3] for frame in frames]

                image_embeddings = []
                cos_sim = []
                for frame in frames:
                    image_embedding = embedding_image(model, processor, frame)

                    image_embedding = normalize_embeddings(image_embedding)
                    cos_sim.append(compute_similarity(text_embeddings, image_embedding).item())

                cos_sim = np.array(cos_sim)
                frame_index = np.linspace(1, len(cos_sim), len(cos_sim))

                figure = plt.figure()
                plt.plot(frame_index, cos_sim )
                plt.xlabel("Frame Index")
                plt.ylabel("Similarity")
                plt.title(f"{task} {diff} {video_idx}")
                # set y axis range [-1,1]
                plt.ylim(-1, 1)

                # plt.savefig(f"progress_img/{env}.png")
                wandb.log({f"progress_video/{task}/{diff}_{video_idx}": wandb.Image(figure)})
                plt.close()
                print(f"progress_video/{task}/{diff}/{video_idx}")

                
                frames = np.stack(frames)
                # predicted_output = np.stack(predicted_output)
                gif_buffer = animate_video_with_rewards(frames, cos_sim, 15)
                
                log_gif_to_wandb(gif_buffer, f"{task}/{diff}_{video_idx}")


    


if __name__ == "__main__":


    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "TrainLIVEvalZeroShot"




    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group="TrainLIVEvalZeroShot",
        name=experiment_name,
    )



    liv_model = load_liv()
    liv_model = liv_model.module
    state_dict = torch.load('/scr/yusenluo/RoboCLIP/LIV/liv/train_liv_roboclip/2024-10-21_05-24-57/snapshot_10000.pt')["liv"]
    liv_model.load_state_dict(state_dict)
    liv_model.eval()
    liv_model = liv_model
    plot_videos("liv", liv_model)


    a = 0