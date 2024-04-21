import h5py
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import os
import tensorflow_datasets as tfds
import random
import numpy as np
import imageio
import cv2
import pandas as pd


# '''
# change annotation at here
# '''
# tar_ann = "pick up the mug on the table."









def main(args):
    df = pd.read_csv('/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv')
    language_model = SentenceTransformer(args.lang_model)
    if 'OpenX' not in df.columns:
        df['OpenX'] = None  # 初始化为空
    for row_index, row in df.iterrows():
    #tar_ann = input("Enter your target annotation: ")
        tar_ann = row['text_label']

        target_latent = language_model.encode(tar_ann, convert_to_tensor=True).squeeze().cpu().numpy()
        h5_file = args.dataset_name + "_lang_emb.h5"
        h5_path = os.path.join(args.dataset_path, h5_file)

        dist = dict()

        with h5py.File(h5_path, 'r') as file:
            anns = list(file.keys())
            random.shuffle(anns)
            use_len = int(np.floor(len(anns) * args.ratio))
            anns = anns[:use_len]

            for ann in tqdm(anns):
                if list(file[ann].keys())[0] == 'index' and list(file[ann].keys())[1] == 'latent':
                    latent = np.asarray(file[ann]["latent"])
                    index = file[ann]["index"][()]
                    use_pair = [ann, index] # annotation, index in the dataset
                    l2_dis = np.linalg.norm(latent - target_latent)
                    if len(dist.keys()) < args.top_k:
                        dist[l2_dis] = use_pair
                    else:
                        max_dis = np.max(list(dist.keys()))
                        if l2_dis < max_dis:
                            dist.pop(max_dis)
                            dist[l2_dis] = use_pair

        print("top", args.top_k, "annotations")
        for ann in dist.keys():
            print(dist[ann][0])
        print("\n")


        # read video
        # Access the 5th data point
        # index = 5
        # data = next(iter(dataset.skip(index).take(1)))
        openx_ds = tfds.load(args.dataset_name, data_dir=args.dataset_path, split=args.split)

        for key in dist.keys():
            pair = dist[key]
            ann = pair[0]
            index = pair[1]

            data = next(iter(openx_ds.skip(index).take(1).prefetch(buffer_size=args.cpus)))
            # data = next(iter(openx_ds.skip(index).take(1).map(lambda x: x, num_parallel_calls=32)))

            # sample = data
            if args.dataset_name in ["droid_100", "droid"]:
                left_1 = list()
                indices = np.linspace(0, len(data["steps"]) - 1, args.ds_frames, dtype=int)
                # print(indices)
                    
                for j, step in enumerate(data["steps"]):
                    if j in indices:
                        img1 = step["observation"]["exterior_image_1_left"].numpy(),
                        left_1.append(img1)
                left_1 = np.squeeze(np.stack(left_1))

                save_path = args.save_folder
                dataset_path = os.path.join(save_path, args.dataset_name)
                left_1_path = os.path.join(dataset_path, "left_1")

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                if not os.path.exists(left_1_path):
                    os.makedirs(left_1_path)

                if args.format == "gif":
                    file_name = ann.replace(" ", "_") + ".gif"
                    df.at[row_index, 'OpenX'] = ann
                    df.to_csv('/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv', index=False)
                    save_1 = os.path.join(left_1_path, file_name)
                    imageio.mimsave(save_1, left_1, fps=args.fps)


                else:
                    if args.format == 'mp4':
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        save_1 = ann.replace(" ", "_") + ".mp4"
                        save_1 = os.path.join(left_1_path, save_1)

                    elif args.format == 'webm':
                        fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM codec
                        save_1 = ann.replace(" ", "_") + ".webm"
                        save_1 = os.path.join(left_1_path, save_1)


                    elif args.format == 'avi':
                        # fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
                        fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
                        save_1 = ann.replace(" ", "_") + ".avi"
                        save_1 = os.path.join(left_1_path, save_1)


                    else:
                        raise ValueError("Invalid video format choice. Choose 'gif', 'mp4', 'avi', or 'webm'.")
                    width, height = left_1.shape[1], left_1.shape[2]
                    # out = cv2.VideoWriter(save_1, fourcc, args.fps, (width, height), True)
                    out = cv2.VideoWriter(save_1, fourcc, args.fps, (height, width), True)

                    # indices = np.linspace(0, left_1.shape[0] - 1, args.ds_frames, dtype=int)

                    for frame_idx in range(left_1.shape[0]):
                        frame = left_1[frame_idx]
                        if args.format == 'mp4' or args.format == 'avi':
                            # Convert from RGB to BGR if saving as MP4
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame)

                    # Release the video writer
                    out.release()

            else:
                imgs = list()
                indices = np.linspace(0, len(data["steps"]) - 1, args.ds_frames, dtype=int)
                for j, step in enumerate(data["steps"]):
                    if j in indices:
                        img = step["observation"]["image"].numpy(),
                        imgs.append(img)
                imgs = np.squeeze(np.stack(imgs))

                save_path = args.save_folder
                dataset_path = os.path.join(save_path, args.dataset_name)
                # video_ffolder = tar_ann.replace(" ", "_")
                # video_path = os.path.join(dataset_path, video_ffolder)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                # if not os.path.exists(video_path):
                #     os.makedirs(video_path)

                if args.format == "gif":
                    file_name = ann.replace(" ", "_") + ".gif"
                    video_path = os.path.join(dataset_path, file_name)
                    imageio.mimsave(video_path, imgs, fps=args.fps)

                else:
                    if args.format == 'mp4':
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        file_name = ann.replace(" ", "_") + ".mp4"
                        video_path = os.path.join(video_path, file_name)
                    elif args.format == 'webm':
                        fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM codec
                        file_name = ann.replace(" ", "_") + ".webm"
                        video_path = os.path.join(video_path, file_name)

                    elif args.format == 'avi':
                        # fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
                        fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
                        file_name = ann.replace(" ", "_") + ".avi"
                        video_path = os.path.join(video_path, file_name)


                    else:
                        raise ValueError("Invalid video format choice. Choose 'gif', 'mp4', or 'webm'.")
                    width, height = imgs.shape[1], imgs.shape[2]

                    out = cv2.VideoWriter(video_path, fourcc, args.fps, (height, width), True)

                    # indices = np.linspace(0, imgs.shape[0] - 1, args.ds_frames, dtype=int)
                    for frame_idx in range(imgs.shape[0]):
                        frame = imgs[frame_idx]
                        if args.format == 'mp4' or args.format == 'avi':
                            # Convert from RGB to BGR if saving as MP4
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame)
                    # Release the video writer
                    out.release()
        #df.to_csv('/scr/yusenluo/RoboCLIP/visualization/video_text_data.csv', index=False)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_model", type=str, default="all-MiniLM-L6-v2", help='language model type')
    parser.add_argument("--dataset_path", type=str, default="/scr/jzhang96/", help='h5 saved location')
    parser.add_argument("--top_k", type=int, default=1, help='show top k close videos')
    parser.add_argument("--ratio", type=float, default=1.0, help='use percentage of data to retrieve video')
    parser.add_argument("--dataset_name", type=str, default="droid", choices=["droid_100", "droid", "bridge", "fractal"], help='dataset saved location')
    parser.add_argument("--split", type=str, default='train', help='training set or test set')
    parser.add_argument("--save_folder", type=str, default='sample_video', help='save_folder')
    parser.add_argument("--fps", type=int, default=30, help='fps of video')
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "webm", "avi"], help='save video format')
    parser.add_argument("--ds_frames", type=int, default=32, help='downsample frames')
    parser.add_argument("--cpus", type=int, default=64, help='num of cpus to use')
    args = parser.parse_args()

    main(args)