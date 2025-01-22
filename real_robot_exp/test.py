import h5py
from tqdm import tqdm
dataset_info = dict()
key_list = ["lang_embedding", "lang_embedding_individual"]
file_path = "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable_processed.h5"
h5_file = h5py.File(file_path, 'a')
key_num = len(h5_file.keys())
dataset_info["key_num"] = key_num
total_num = 0
max_length = 0
min_length = 1000000
shorter_length = 0
# del h5_file['Place red']
total_length = []
for key in tqdm(h5_file.keys()):
    data_group = h5_file[key]
    num = len(data_group) - 1
    total_num += num
    for sub_key in data_group.keys():
        if "lang_embedding" not in list(data_group.keys()):
            print(key, sub_key, "no lang_embedding")
            del data_group[sub_key]
        else:
            if sub_key not in key_list:
                length = len(data_group[sub_key])
                if length <= 5:
                    print(key, sub_key, "length <= 5", length)
                    # print(key, sub_key)
                    del data_group[sub_key]
                    # import pdb; pdb.set_trace()
                    if "lang_embedding" in list(data_group.keys()) and "lang_embedding_individual" in list(data_group.keys()):
                        if len(data_group) == 2:
                            print(key, "delete")
                            del h5_file[key]
                    elif "lang_embedding" in list(data_group.keys()):
                        if len(data_group) == 1:
                            print(key, "delete")
                            del h5_file[key]
                    
                    shorter_length += 1
                if length > max_length:
                    max_length = length
                if length < min_length:
                    min_length = length
                total_length.append(length)

dataset_info["total_num"] = total_num
dataset_info["max_length"] = max_length
dataset_info["min_length"] = min_length
dataset_info["shorter_length"] = shorter_length
# dataset_info["avg_length"] = sum(total_length) / len(total_length)
# dataset_info["median_length"] = sorted(total_length)[len(total_length) // 2]
print(dataset_info)

h5_file.close()
import json
with open("openx_dataset_info_new.json", "w") as f:
    json.dump(dataset_info, f, indent=4)