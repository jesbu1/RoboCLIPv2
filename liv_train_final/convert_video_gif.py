import os 
import cv2
import imageio
from tqdm import tqdm


base_path = "/home/jzhang96/RoboCLIPv2/liv_train_final/self_collected_vids"
og_video_names = ["1.mp4", "2.mp4", "3.mp4", "4.mp4", "5.mp4"]
folder_names = os.listdir(base_path)
for folder_name in tqdm(folder_names):
    folder_path = os.path.join(base_path, folder_name)
    category_names = os.listdir(folder_path)
    for category_name in category_names:
        category_path = os.path.join(folder_path, category_name)
        video_names = os.listdir(category_path)
        for video_name in video_names:
            if video_name.endswith(".mp4"):
                video_path = os.path.join(category_path, video_name)
                gif_path = os.path.join(category_path, video_name.replace(".mp4", ".gif"))

                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:  # Break if no frames are left
                        break
                    # Convert the frame from BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # center crop 224x224
                    h, w, _ = rgb_frame.shape
                    start_h = (h - 224) // 2
                    start_w = (w - 224) // 2
                    rgb_frame = rgb_frame[start_h:start_h + 224, start_w:start_w + 224]
                    frames.append(rgb_frame)

                duration = 1 / 30
                imageio.mimsave(gif_path, frames, duration=duration)

                # if video_name not in og_video_names:
                #     if "1.mp4" not in os.listdir(category_path):
                #         os.rename(video_path, os.path.join(category_path, "1.mp4"))
                #         video = os.path.join(category_path, "1.mp4")
                #         gif = os.path.join(category_path, "1.gif")
                #         iio.mimwrite(gif, iio.get_reader(video).__iter__())

                #     else:
                #         os.rename(video_path, os.path.join(category_path, "2.mp4"))




                    




