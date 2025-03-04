#!/bin/bash

# Base URL of the TFDS dataset
BASE_URL="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/"

# Directory to save the downloaded files
OUTPUT_DIR="/data/shared/openx_rlds_data/bridge_v2"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Use wget to recursively download all files from the directory
wget --no-check-certificate -r -np -nH --cut-dirs=5 -R "index.html*" -P "$OUTPUT_DIR" "$BASE_URL"

echo "Download completed. Files are saved in $OUTPUT_DIR."


