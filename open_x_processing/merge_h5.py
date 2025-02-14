from tqdm import tqdm
import h5py

#MAIN_H5 = "/data/shared/roboclip/data/h5_buffers/openx_embeddings/openx_embeddings_full_uncompressed_with_langtable35k.h5"
MAIN_H5 = "full_openx_embeddings_dino_train.h5"
H5_MERGING_FROM = "droid_embeddings_dino.h5"
print(f"Adding to {MAIN_H5} from {H5_MERGING_FROM}")
MAX_TO_MERGE = 30_000


# make a set to keep track of the tasks we've seen
total_samples = 0
with h5py.File(MAIN_H5, "a") as f:
    num_keys_before = len(f.keys())
    num_trajs_before = sum(
        [len([key for key in f[task].keys() if key.isnumeric()]) for task in f.keys()]
    )
    with h5py.File(H5_MERGING_FROM, "r") as f2:
        for task in tqdm(f2.keys()):
            if total_samples >= MAX_TO_MERGE:
                break
            if task in f.keys():
                # get all keys that are numeric
                all_f_keys_for_task = list(f[task].keys())
                all_f_keys_for_task = [
                    int(k) for k in all_f_keys_for_task if k.isnumeric()
                ]
                next_key = max(all_f_keys_for_task) + 1
                for key in f2[task].keys():
                    if key.isnumeric():
                        f[task].create_dataset(
                            str(next_key),
                            data=f2[task][key],
                        )
                        next_key += 1
                        total_samples += 1
            else:
                f.create_group(task)
                for key in f2[task].keys():
                    f[task].create_dataset(
                        key,
                        data=f2[task][key],
                    )
                    total_samples += 1
    num_keys_after = len(f.keys())
    num_trajs_after = sum(
        [len([key for key in f[task].keys() if key.isnumeric()]) for task in f.keys()]
    )
    print(f"Added {num_keys_after - num_keys_before} extra tasks")
    print(
        f"Added {num_trajs_after - num_trajs_before} extra trajectories (includes datasets not corresponding to trajs)"
    )
