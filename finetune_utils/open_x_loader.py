from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import os
import cv2
import torch


def adjust_size(frames):
    """
    Adjusts the size of the frames to a target height and width by cropping.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Cropped frames as a numpy array.
    """
    if len(frames) == 0:
        return np.array([])

    target_height = 224
    target_width = 224

    height, width, _ = frames[0].shape

    if height < target_height or width < target_width:
        adjusted_frames = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR) for frame in
                           frames]
    else:
        start_x = width // 2 - target_width // 2
        start_y = height // 2 - target_height // 2
        adjusted_frames = [frame[start_y:start_y + target_height, start_x:start_x + target_width] for frame in frames]

    return np.array(adjusted_frames)





class TextVideoDataset(Dataset):
    def __init__(self, args, h5_file, split_file, split="train"):
        if args.debug:
            split_file = {
                "train": list(h5_file.keys()),
                "val": list(h5_file.keys()),
            }

        self.args = args
        self.ann_key = ["ann_1", "ann_2", "ann_3"]
        self.preprocess = args.preprocess
        self.ds_frames = args.ds_frames
        self.keys = split_file[split]
        self.h5 = h5_file
        

    def __len__(self):
        # return 50
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]
        
        if self.args.dataset_name in ["droid_100", "droid"]:
            # video_frames = self.h5[key]["left_1"][()]
            video_frames = np.asarray(self.h5[key]["left_1"])
            # video_frames = video_frames["left_1"][()]

            video_frames = np.array(video_frames)
            indices = np.linspace(0, len(video_frames) - 1, self.ds_frames, dtype=int)
            video_frames = video_frames[indices]
            if self.preprocess:
                video_frames = video_frames / 255.0
                # resize video frames to 224x224
                video_frames = adjust_size(video_frames)
            
            texts = [self.h5[key][ann_key][()].decode() for ann_key in self.ann_key]
            if not self.args.debug:
                texts = [text for text in texts if text != ""]
            # # random choice from texts
            text = random.choice(texts)

        else:
            video_frames = self.h5[key]["video"][()]
            video_frames = np.array(video_frames)
            indices = np.linspace(0, len(video_frames) - 1, self.ds_frames, dtype=int)
            video_frames = video_frames[indices]

            text = self.h5[key]["ann"][()].decode()
            
        return {"video_frames": video_frames, "text": text}



class XCLIPDataset(TextVideoDataset):
    def __init__(self, args, h5_file, split_file, split="train", tokenizer = None):
        if args.debug:
            split_file = {
                "train": list(h5_file.keys()),
                "val": list(h5_file.keys()),
            }

        self.args = args
        self.ann_key = ["ann_1", "ann_2", "ann_3"]
        self.preprocess = args.preprocess
        self.ds_frames = args.ds_frames
        self.keys = split_file[split]
        self.h5 = h5_file
        self.tokenizer = tokenizer
        

    def __len__(self):
        # return 50
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]
        
        if self.args.dataset_name in ["droid_100", "droid"]:
            # video_frames = self.h5[key]["left_1"][()]
            video_frames = np.asarray(self.h5[key]["left_1"])
            # video_frames = video_frames["left_1"][()]

            video_frames = np.array(video_frames)
            indices = np.linspace(0, len(video_frames) - 1, self.ds_frames, dtype=int)
            video_frames = video_frames[indices]
            if self.preprocess:
                video_frames = video_frames / 255.0
                # resize video frames to 224x224
                video_frames = adjust_size(video_frames)
            
            texts = [self.h5[key][ann_key][()].decode() for ann_key in self.ann_key]
            if not self.args.debug:
                texts = [text for text in texts if text != ""]
            # # random choice from texts
            text = random.choice(texts)

        else:
            video_frames = self.h5[key]["video"][()]
            video_frames = np.array(video_frames)
            indices = np.linspace(0, len(video_frames) - 1, self.ds_frames, dtype=int)
            video_frames = video_frames[indices]

            text = self.h5[key]["ann"][()].decode()

        # text_emb = self.tokenizer(text, padding=True, return_tensors="pt")
        # video_emb = self.processor(videos=video_frames, return_tensors="pt")
            
        return {"video_frames": video_frames, "text": text}




class EmbeddingsDataset(Dataset):
    def __init__(self, video_embeddings, text_embeddings, video_ids, text_labels):
        self.video_embeddings = video_embeddings
        self.text_embeddings = text_embeddings
        self.video_ids = video_ids
        self.text_labels = text_labels

    def __len__(self):
        return len(self.video_embeddings)

    def __getitem__(self, idx):
        video_embedding = self.video_embeddings[idx]
        text_embedding = self.text_embeddings[idx]
        video_id = self.video_ids[idx]
        text_label = self.text_labels[idx]
        return {"video_embedding": video_embedding,
                "text_embedding": text_embedding,
                "video_id": video_id,
                "text_label": text_label}


