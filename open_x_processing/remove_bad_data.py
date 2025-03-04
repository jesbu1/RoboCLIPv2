import h5py
import os
import argparse
import numpy as np
from tqdm import tqdm

# Constants
DINO_EMBEDDING_DIM = 768
LANG_EMBEDDING_KEYS = [
    "liv_lang_embedding",
    "minilm_lang_embedding",
    "liv_lang_embedding_individual",
    "minilm_lang_embedding_individual",
]
BAD_INSTRUCTIONS = {
    "no action",
    "none",
    "no instruction",
    "no task",
    "no command",
    "",
    " ",
    ".",
    "no language instruction",
    "no natural language instruction",
}


def is_valid_embedding(data):
    """Check if the embedding has the correct shape."""
    return data.shape[-1] == DINO_EMBEDDING_DIM


def is_good_instruction(task_name):
    """Check if the instruction is valid."""
    task_lower = task_name.lower().strip()
    # Check if instruction is in bad list
    if task_lower in BAD_INSTRUCTIONS:
        return False
    # Check if instruction is too short
    if len(task_lower) <= 4:
        return False
    return True


def clean_h5_file(input_path, output_path=None):
    """
    Clean an H5 file by removing bad data and validating embeddings.

    Args:
        input_path (str): Path to input H5 file
        output_path (str): Path to output H5 file. If None, will modify input file
    """
    if output_path is None:
        output_path = input_path + ".tmp"
        need_rename = True
    else:
        need_rename = False

    print(f"Processing {input_path}")
    stats = {
        "tasks_removed": 0,
        "embeddings_removed": 0,
        "total_tasks": 0,
        "total_embeddings": 0,
    }

    with h5py.File(input_path, "r") as in_f, h5py.File(output_path, "w") as out_f:
        # Iterate through all tasks
        for task_name in tqdm(in_f.keys(), desc="Processing tasks"):
            stats["total_tasks"] += 1

            # Skip bad instructions
            if not is_good_instruction(task_name):
                stats["tasks_removed"] += 1
                continue

            task_group = in_f[task_name]
            valid_datasets = []

            # Check all datasets in the task
            for dataset_name in task_group.keys():
                stats["total_embeddings"] += 1
                try: 
                    data = task_group[dataset_name][:]
                except Exception as e:
                    stats["tasks_removed"] += 1
                    print(f"Exception {e}, skipping this datapoint. dataset_name: {dataset_name}, task group keys: {task_group.keys(), task_name: {task_name}")
                    continue

                # Skip validation for language embeddings
                if any(x in dataset_name for x in LANG_EMBEDDING_KEYS):
                    valid_datasets.append((dataset_name, data))
                    continue

                # Validate image embeddings
                if is_valid_embedding(data):
                    valid_datasets.append((dataset_name, data))
                else:
                    stats["embeddings_removed"] += 1
                    print(
                        f"Removing invalid embedding {dataset_name} in task {task_name} with shape {data.shape}"
                    )

            # If we have valid datasets, create the task group and save them
            if valid_datasets and len(valid_datasets) > len(LANG_EMBEDDING_KEYS):
                out_task_group = out_f.create_group(task_name)
                for dataset_name, data in valid_datasets:
                    out_task_group.create_dataset(dataset_name, data=data)

    # Print statistics
    print("\nCleaning Statistics:")
    print(f"Total tasks processed: {stats['total_tasks']}")
    print(f"Tasks removed: {stats['tasks_removed']}")
    print(f"Total embeddings processed: {stats['total_embeddings']}")
    print(f"Invalid embeddings removed: {stats['embeddings_removed']}")

    if need_rename:
        os.replace(output_path, input_path)
        print(f"\nUpdated original file: {input_path}")
    else:
        print(f"\nSaved cleaned data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean H5 files by removing bad data")
    parser.add_argument(
        "input_path", type=str, help="Path to input H5 file or directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output H5 file (only used if input is a single file)",
    )

    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        # Process all H5 files in directory
        for filename in os.listdir(args.input_path):
            if filename.endswith(".h5"):
                file_path = os.path.join(args.input_path, filename)
                clean_h5_file(file_path)
    else:
        # Process single file
        clean_h5_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
