import argparse
import tensorflow_datasets as tfds
from tqdm import tqdm
import h5py
import os
import numpy as np



'''
save the dataset into text-video pair in h5 format

'''



# def main(args):



#     ds = tfds.load(args.dataset_name, data_dir=args.dataset_path, split=args.split)

#     num = 0
#     total = 0


#     # initialize a new h5 file


#     for i, example in tqdm(enumerate(ds)):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`

#         if args.dataset_name in ["droid_100", "droid"]:
#             video_frames_1 = []
#             video_frames_2 = []
#             for j, step in enumerate(example["steps"]):
#                 if j == 0:
#                     key1 = step["language_instruction"].numpy().decode()
#                     key2 = step["language_instruction_2"].numpy().decode()
#                     key3 = step["language_instruction_3"].numpy().decode()
#                 if j == 0:
#                     break
#             total += 1
#             if key1 == "" and key2 == "" and key3 == "":
#                 num += 1
#             print("total", total, "empty", num)
#         else:

#             for j, step in enumerate(example["steps"]):
#                 key = step['observation']['natural_language_instruction'].numpy().decode()
#                 if j == 0:
#                     break
#             total += 1
#             if key == "":
#                 num += 1

#             print("total", total, "empty", num, "key", key)
                





# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lang_model", type=str, default="all-MiniLM-L6-v2", help='language model type')
#     parser.add_argument("--dataset_path", type=str, default="/scr/jzhang96/", help='dataset saved location')
#     parser.add_argument("--dataset_name", type=str, default="droid", choices=["droid_100", "droid", "bridge", "fractal"], help='dataset saved location')
#     parser.add_argument("--split", type=str, default='train', help='training set or test set')

#     args = parser.parse_args()

#     main(args)


path = "/scr/jzhang96/droid_torch_compress_train.h5"
f = h5py.File(path, 'r')
print(len(f.keys()))

