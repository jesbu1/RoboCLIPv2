import imageio
from PIL import Image
import os

# Function to center crop an image
def center_crop(image, crop_size):
    width, height = image.size
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return image.crop((left, top, right, bottom))


# Load the gif using imageio
base_path = "/home/jzhang96/RoboCLIPv2/clip/reward_eval_videos"
tasks = ["button_press_wall", "topdown", "windowclose"]
idxs = ["1.gif", "2.gif"]

for task in tasks:
    task_path = os.path.join(base_path, task, "GT")

    for idx in idxs:
        gif_path = os.path.join(task_path, idx)

        gif = imageio.mimread(gif_path)

        # Define center crop dimensions
        crop_size = (224, 224)

        # Process each frame in the GIF
        cropped_frames = []
        for frame in gif:
            # Convert frame to PIL Image for cropping
            pil_image = Image.fromarray(frame)
            cropped_image = center_crop(pil_image, crop_size)
            cropped_frames.append(cropped_image)

        # Save the cropped gif
        # cropped_gif_path = '1_cropped.gif'
        cropped_frames[0].save(gif_path, save_all=True, append_images=cropped_frames[1:], loop=0)

        print(f"Saved cropped gif as {gif_path}")

