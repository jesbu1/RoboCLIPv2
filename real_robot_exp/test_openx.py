import h5py
from tqdm import tqdm
import numpy as np
dataset_info = dict()

file_path = "/data/shared/roboclip/data/h5_buffers/openx_embeddings/full_openx_embeddings_dino_test_backup.h5"
h5_file = h5py.File(file_path, 'a')
key_num = len(h5_file.keys())
dataset_info["key_num"] = key_num
total_num = 0
max_length = 0
min_length = 1000000
shorter_length = 0
# del h5_file['Place red
num = 0
del_num = 0
for key in tqdm(h5_file.keys()):
    data_group = h5_file[key]
    if len(data_group) <= 4:
        print(key)
        del h5_file[key]
    else:
        for sub_key in data_group.keys():
            if "lang" not in sub_key:
                data = np.asarray(data_group[sub_key])
                if len(data) < 5:
                    print(key, sub_key)
                    num += 1
                    del data_group[sub_key]
                    if len(data_group) == 4:
                        del_num += 1
                        del h5_file[key]
                

            #     print(key, sub_key)
            #     num += 1
            #     del data_group[sub_key]
            #     if len(data_group) == 4:
            #         del_num += 1
            #         del h5_file[key]

print(num, del_num)





    

print(num)

    