import transformers
import torch
import h5py
import numpy as np

h5_file = h5py.File("/scr/jzhang96/metaworld_25_for_clip_liv.h5", "r")
data = h5_file['liv']['sweep-into-v2_text']
data = np.array(data)
import pdb; pdb.set_trace()
print(data.shape)
norm = np.linalg.norm(data, axis=1)
print(norm)
data = h5_file['liv']["coffee-push-v2"]['1']

data = np.array(data)

norm = np.linalg.norm(data, axis=1)
print(norm)

