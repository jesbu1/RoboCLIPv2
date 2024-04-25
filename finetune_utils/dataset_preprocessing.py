import argparse
import tensorflow_datasets as tfds
from tqdm import tqdm
import h5py
import os
import numpy as np



'''
save the dataset into text-video pair in h5 format

'''



def main(args):



    ds = tfds.load(args.dataset_name, data_dir=args.dataset_path, split=args.split)

    num = 0


    # initialize a new h5 file
    h5_file = args.dataset_name + "_torch_compress_test_" + args.split + ".h5"
    h5_path = os.path.join(args.dataset_path, h5_file)
    h5 = h5py.File(h5_path, 'w')

    for i, example in tqdm(enumerate(ds)):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`

        if args.dataset_name in ["droid_100", "droid"]:
            video_frames_1 = []
            video_frames_2 = []
            for j, step in enumerate(example["steps"]):
                if j == 0:
                    key1 = step["language_instruction"].numpy().decode()
                    key2 = step["language_instruction_2"].numpy().decode()
                    key3 = step["language_instruction_3"].numpy().decode()

                img1 = step["observation"]["exterior_image_1_left"].numpy()
                # img2 = step["observation"]["exterior_image_2_left"].numpy()
                video_frames_1.append(img1)
                # video_frames_2.append(img2)

            if key1 == "" and key2 == "" and key3 == "":
                continue
                
            h5_dataset_name = str(i)
            h5.create_group(h5_dataset_name)
            # five keys: "ann_1", "ann_2", "ann_3", "left_1", "left_2"
            h5[h5_dataset_name].create_dataset("ann_1", data=key1)
            h5[h5_dataset_name].create_dataset("ann_2", data=key2)
            h5[h5_dataset_name].create_dataset("ann_3", data=key3)
            # h5[h5_dataset_name].create_dataset("left_1", data=video_frames_1, compression='gzip', compression_opts=9)
            h5[h5_dataset_name].create_dataset("left_1", data=video_frames_1)

            # h5[h5_dataset_name].create_dataset("left_2", data=video_frames_2)

        # if read anotation: h5[h5_dataset_name]["ann_1"][()].decode()
        # if read video: h5[h5_dataset_name]["left_1"][()]

        # test video:
        # import imageio
        # imageio.mimsave("save_2.gif", video_frames_2, fps=30)


        else:
        #     # "fractal, bridge": "natural_language_instruction"
            video_frames = []

            for j, step in enumerate(example["steps"]):
                if j == 0:
                    ann = step["observation"]["natural_language_instruction"].numpy().decode()
                img = step["observation"]["image"].numpy()
                video_frames.append(img)

            if ann == "":
                continue
            h5_dataset_name = str(i)
            h5.create_group(h5_dataset_name)
            h5[h5_dataset_name].create_dataset("ann", data=ann)
            h5[h5_dataset_name].create_dataset("video", data=video_frames)
            # h5[h5_dataset_name].create_dataset("video", data=video_frames, compression='gzip', compression_opts=9)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_model", type=str, default="all-MiniLM-L6-v2", help='language model type')
    parser.add_argument("--dataset_path", type=str, default="/scr/jzhang96/", help='dataset saved location')
    parser.add_argument("--dataset_name", type=str, default="droid_100", choices=["droid_100", "droid", "bridge", "fractal"], help='dataset saved location')
    parser.add_argument("--split", type=str, default='train', help='training set or test set')

    args = parser.parse_args()

    main(args)



