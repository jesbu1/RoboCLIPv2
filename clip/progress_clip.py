import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import imageio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt



video_path = "/scr/jzhang96/reward_eval_videos/"
tasks = ["topdown", "windowclose", "button_press_wall"]
hardnesses = ["all_fail", "close_fail", "success", "GT"]
idxs = ["1", "2"]
annotation = ["pressing button", "closing window", "pressing button"]


# Load CLIP model and processor (for ViT-L/14)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Example text and image input
text = "Robot closing a door"

hardnesses = ["all_fail", "close_fail", "success", "GT"]
for hardness in hardnesses:
    video_path = f"/home/jzhang96/RoboCLIPv2/losses/reward_eval_videos/windowclose/{hardness}/1.gif"
    frames = imageio.mimread(video_path)
    frames = [frame[:,:,:3] for frame in frames]
    inputs = processor(text=[text] * len(frames), images=frames, return_tensors="pt", padding=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    text_embeddings = outputs.text_embeds  # Shape: (batch_size, 768)
    image_embeddings = outputs.image_embeds  # Shape: (batch_size, 768)

    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    cosine_similarity = F.cosine_similarity(text_embeddings, image_embeddings)

    frame_index = np.linspace(0, len(cosine_similarity) - 1, len(cosine_similarity)) + 1
    plt.figure()
    plt.plot(frame_index, cosine_similarity.detach().cpu().numpy())
    plt.xlabel("Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Window Close {hardness}")
    # save as png then close figure
    plt.savefig(f"progress_img/clip_window_close_{hardness}.png")
    plt.close()

