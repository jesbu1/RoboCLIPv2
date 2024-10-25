import clip
import torch
import torchvision.transforms as T
from PIL import Image 
import imageio
from liv import load_liv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

# loading LIV
liv = load_liv()
liv.eval()

text = clip.tokenize(["robot close window"]).to(device)
text_embedding = liv(input=text, modality="text")
# compute LIV image and text embedding

transform = T.Compose([T.ToTensor()])


## ENCODE IMAGE

hardnesses = ["all_fail", "close_fail", "success", "GT"]
for hardness in hardnesses:
    video_path = f"/home/jzhang96/RoboCLIPv2/losses/reward_eval_videos/windowclose/{hardness}/1.gif"
    frames = imageio.mimread(video_path)
    frames = [frame[:,:,:3] for frame in frames]
    sim_list = []
    for frame in frames:
        frame = transform(Image.fromarray(frame.astype(np.uint8))).unsqueeze(0).to(device)
        with torch.no_grad():
            img_embedding = liv(input=frame, modality="vision")

        # compute LIV value
        img_text_value = liv.module.sim(img_embedding, text_embedding)
        # Output: [ 0.1151, -0.0151, -0.0997]
        sim_list.append(img_text_value.item())

    frame_index = np.linspace(0, len(sim_list) - 1, len(sim_list)) + 1
    plt.figure()
    plt.plot(frame_index, sim_list)
    plt.xlabel("Frame Index")
    plt.ylabel("LIV Value")
    plt.title(f"Window Close {hardness}")
    # save as png then close figure
    plt.savefig(f"progress_img/liv_window_close_{hardness}.png")
    print(f"Saved {hardness} to liv_window_close_{hardness}.png")
    plt.close()

