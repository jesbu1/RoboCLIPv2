import h5py
import os
import random
# h5_path = os.path.join("/scr/jzhang96/", "droid_torch_compress_train.h5") 
# # h5_path = os.path.join("/scr/jzhang96/", "droid_100_torch_compress_train.h5") 

# print(h5_path)
# h5 = h5py.File(h5_path, 'r')
# keys = list(h5.keys())
# random.shuffle(keys)
# train_keys = keys[:int(len(keys)*0.95)]
# val_keys = keys[int(len(keys)*0.95):]
# dicts = {}
# dicts["train"] = train_keys
# dicts["val"] = val_keys
# print("train", len(train_keys))
# print("val", len(val_keys))
# import json
# with open("finetune_utils/droid_torch_compress_train_val.json", "w") as f:
#     json.dump(dicts, f)  
import numpy as np
# load s3d_dict.npy
s3d_dict = np.load("s3d_dict.npy", allow_pickle=True)
import pdb ; pdb.set_trace()
