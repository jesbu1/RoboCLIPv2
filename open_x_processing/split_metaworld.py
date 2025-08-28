from new_task_annotation_v2 import train_gt_annotation, eval_gt_annotation, generated_gt_annotation
import h5py
h5_file_name = "metaworld_dino_embeddings_224_fix.h5"
h5_file = h5py.File(h5_file_name, 'r')
train_keys = list(train_gt_annotation.keys())
eval_keys = list(eval_gt_annotation.keys())
print(len(train_keys), len(eval_keys))


h5_train_file = h5py.File("metaworld_dino_embeddings_train_fix.h5", 'w')
h5_eval_file = h5py.File("metaworld_dino_embeddings_eval_fix.h5", 'w')

for key in train_keys:
    if key not in h5_file:
        print("Key not found: ", key)
    else:
        print("Adding Key ", key)
        h5_file.copy(key, h5_train_file)

for key in eval_keys:
    if key not in h5_file:
        print("Key not found: ", key)
    else:
        print("Adding Key ", key)
        h5_file.copy(key, h5_eval_file)

import pdb ; pdb.set_trace()

