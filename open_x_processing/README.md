# Open-X Processing Tools

This directory contains tools for processing the Open-X dataset to create and manage embedding files.

## Scripts

### 1. `create_h5.py`

This script processes Open-X datasets and creates H5 files containing embeddings for both images and language instructions.

#### Features
- Processes multiple camera views per episode
- Supports both DINO and LIV embedding models
- Creates embeddings for language instructions using LIV and MiniLM models
- Saves embeddings for each dataset in separate H5 files

#### Configuration
Key parameters at the top of the script:
```python
TFDS_PATH = "/data/shared/openx_rlds_data"  # Path to OpenX dataset
TRAIN_SPLIT = "train"  # Dataset split to process
SAVE_H5_DIR = "dataset_embeddings"  # Output directory for H5 files
DEBUG = False  # If True, only processes 10 samples per dataset
MAX_NUM_FRAMES_PER_EPISODE = 32  # Number of frames to sample per episode
MAX_EPISODES_FOR_LANG_TABLE = 10000  # Limit for language_table dataset
```

#### Usage
```bash
python create_h5.py
```

### 2. `merge_all_h5.py`

This script merges multiple H5 files (created by `create_h5.py`) into a single combined H5 file.

#### Features
- Merges all H5 files from a specified directory
- Maintains unique indices for image embeddings
- Preserves language embeddings without duplication
- Handles errors gracefully

#### Usage
```bash
# Using default values
python merge_all_h5.py

# With custom paths
python merge_all_h5.py --input_dir path/to/h5/files --output_file merged_output.h5
```

#### Arguments
- `--input_dir`: Directory containing H5 files to merge (default: "dataset_embeddings")
- `--output_file`: Path for the output merged file (default: "merged_embeddings.h5")

## File Structure

The H5 files are structured as follows:

```
dataset_name_train_embeddings.h5/
├── task_1/
│   ├── liv_lang_embedding
│   ├── liv_lang_embedding_individual
│   ├── minilm_lang_embedding
│   ├── minilm_lang_embedding_individual
│   ├── 0_primary
│   ├── 0_overhead
│   ├── 1_primary
│   └── ...
├── task_2/
└── ...
```

Where:
- Each task has its own group
- Language embeddings are stored once per task
- Image embeddings are stored as `{index}_{camera_view}`
- Camera views include 'primary', 'overhead', etc. (varies by dataset)

## Requirements

- tensorflow-datasets
- torch
- torchvision
- h5py
- numpy
- PIL
- transformers
- tqdm

## Notes

- The scripts handle large datasets efficiently by processing them in batches
- For the language_table dataset, processing is limited to prevent memory issues
- Error handling ensures the process continues even if individual files or episodes fail
- The merge script automatically handles index conflicts when combining files 