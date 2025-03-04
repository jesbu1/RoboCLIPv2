# Open-X Processing Tools

This directory contains tools for processing the Open-X dataset to create and manage embedding files.

## Full pipeline in short

```bash
python create_h5.py #once for train, once for test by editing top of the file
python create_h5_droid.py
python merge_all_h5.py --input_dir train_dataset_embeddings --output_file merged_train_embeddings.h5
python merge_all_h5.py --input_dir test_dataset_embeddings --output_file merged_test_embeddings.h5
python remove_bad_data.py merged_train_embeddings.h5 --output_path cleaned_merged_train_embeddings.h5
python remove_bad_data.py merged_test_embeddings.h5 --output_path cleaned_merged_test_embeddings.h5
python count_trajectories.py cleaned_merged_train_embeddings.h5 --save_stats train_stats.json
python count_trajectories.py cleaned_merged_test_embeddings.h5 --save_stats test_stats.json
```
### 1. `create_h5.py/create_h5_droid.py`

This script processes Open-X datasets and creates H5 files containing embeddings for both images and language instructions.

#### Features
- Processes multiple camera views per episode
- Supports both DINO and LIV embedding models
- Creates embeddings for language instructions using LIV and MiniLM models
- Saves embeddings for each dataset in separate H5 files

#### Configuration
Key parameters at the top of the script for `create_h5.py`:
```python
TFDS_PATH = "/data/shared/openx_rlds_data"  # Path to OpenX dataset
TRAIN_SPLIT = "train"  # Dataset split to process
SAVE_H5_DIR = "dataset_embeddings"  # Output directory for H5 files
DEBUG = False  # If True, only processes 10 samples per dataset
MAX_NUM_FRAMES_PER_EPISODE = 32  # Number of frames to sample per episode
MAX_EPISODES_FOR_LANG_TABLE = 10000  # Limit for language_table dataset
```

Key parameters at the top of the script for `create_h5_droid.py`:
```python
TRAIN_SPLIT = "train"  # Dataset split to process
SAVE_H5_NAME = "droid_embeddings_dino.h5"  # Name of the output H5 file
DEBUG = False  # If True, only processes 10 samples per dataset
MAX_NUM_FRAMES_PER_EPISODE = 32  # Number of frames to sample per episode
MAX_SAMPLES = float("inf")  # Number of samples to process
```

#### Usage
```bash
python create_h5.py

# also do DROID
python create_h5_droid.py 
mv droid_embeddings_dino.h5 train_dataset_embeddings/
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

# Remove bad data from the merged output
python remove_bad_data.py merged_output.h5 --output_path cleaned_merged_output.h5

# Replace the merged output with the cleaned output
mv cleaned_merged_output.h5 merged_output.h5

# Save statistics to JSON file
python count_trajectories.py merged_output.h5 --save_stats stats.json
```

#### Arguments
- `--input_dir`: Directory containing H5 files to merge (default: "dataset_embeddings")
- `--output_file`: Path for the output merged file (default: "merged_embeddings.h5")

### 3. `remove_bad_data.py`

This script cleans H5 files by removing invalid embeddings and poor quality language instructions.

#### Features
- Validates DINO embedding dimensions (should be 768)
- Removes tasks with poor quality instructions (e.g., "no action", "none")
- Filters out instructions that are too short
- Preserves language embeddings without validation
- Provides detailed statistics about removed data

#### Usage
```bash
# Clean a single file
python remove_bad_data.py input.h5

# Clean a single file and save to new location
python remove_bad_data.py input.h5 --output_path cleaned.h5

# Clean all H5 files in a directory
python remove_bad_data.py path/to/directory
```

#### Arguments
- `input_path`: Path to input H5 file or directory
- `--output_path`: Optional path for output file (only used for single file processing)

### 4. `count_trajectories.py`

This script analyzes H5 files and provides detailed statistics about the number of trajectories.

#### Features
- Counts total number of trajectories across all tasks
- Provides per-task trajectory counts
- Shows distribution across different camera views
- Lists top 10 tasks by number of trajectories
- Can export statistics to JSON for further analysis

#### Usage
```bash
# Basic usage
python count_trajectories.py input.h5

# Save statistics to JSON file
python count_trajectories.py input.h5 --save_stats stats.json
```

#### Arguments
- `input_path`: Path to H5 file to analyze
- `--save_stats`: Optional path to save statistics as JSON

## File Structure

The H5 files are structured as follows:

```
dataset_name_train_embeddings.h5/
├── task_1/
│   ├── liv_lang_embedding
│   ├── liv_lang_embedding_individual
│   ├── minilm_lang_embedding
│   ├── minilm_lang_embedding_individual
│   ├── 4
│   ├── 5
│   ├── 6
│   └── ...
├── task_2/
└── ...
```

Where:
- Each task has its own group
- Language embeddings are stored once per task
- Image embeddings are stored as `{index}`
- Camera views include 'primary', 'overhead', etc. (varies by dataset). They're merged into the same keys.

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
- The cleaning script provides statistics about removed data and invalid embeddings 
