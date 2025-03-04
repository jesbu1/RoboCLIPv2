import h5py
import os
from tqdm import tqdm
import argparse


def merge_h5_files(input_dir, output_file):
    """
    Merge all H5 files in the input directory into a single H5 file.

    Args:
        input_dir (str): Directory containing H5 files to merge
        output_file (str): Path to the output merged H5 file
    """
    # Get list of all H5 files in the input directory
    h5_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    print(f"Found {len(h5_files)} H5 files to merge")

    # Create output H5 file
    with h5py.File(output_file, "w") as out_f:
        # Iterate through each input H5 file
        for h5_file in tqdm(h5_files, desc="Merging H5 files"):
            file_path = os.path.join(input_dir, h5_file)
            with h5py.File(file_path, "r") as in_f:
                # Get all task groups in the input file
                for task_name in in_f.keys():
                    # Create task group in output file if it doesn't exist
                    if task_name not in out_f:
                        task_group = out_f.create_group(task_name)
                        # Copy all datasets from input task group to output task group
                        for dataset_name in in_f[task_name].keys():
                            in_f[task_name].copy(dataset_name, task_group)
                    else:
                        # If task exists, append new datasets with updated indices
                        task_group = out_f[task_name]
                        try:
                            current_max_idx = max(
                                [
                                    int(k.split("_")[0])
                                    for k in task_group.keys()
                                    if k
                                    not in [
                                        "liv_lang_embedding",
                                        "liv_lang_embedding_individual",
                                        "minilm_lang_embedding",
                                        "minilm_lang_embedding_individual",
                                    ]
                                ],
                                default=-1,
                            )

                            for dataset_name in in_f[task_name].keys():
                                # Skip language embeddings as they should be the same for each task
                                if dataset_name in [
                                    "liv_lang_embedding",
                                    "liv_lang_embedding_individual",
                                    "minilm_lang_embedding",
                                    "minilm_lang_embedding_individual",
                                ]:
                                    continue

                                # Update index for image embeddings
                                idx, img_key = dataset_name.split("_", 1)
                                new_name = f"{current_max_idx + 1 + int(idx)}_{img_key}"
                                in_f[task_name].copy(
                                    dataset_name, task_group, name=new_name
                                )
                        except Exception as e:
                            print(f"Encountered exception {e} with task group keys: {task_group.keys()}, task name: {task_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple H5 files into a single file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset_embeddings",
        help="Directory containing H5 files to merge",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="merged_embeddings.h5",
        help="Path to output merged H5 file",
    )

    args = parser.parse_args()

    print(f"Merging H5 files from {args.input_dir} into {args.output_file}")
    merge_h5_files(args.input_dir, args.output_file)
    print("Merge complete!")


if __name__ == "__main__":
    main()
