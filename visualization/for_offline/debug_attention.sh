#!/bin/bash
#SBATCH --job-name=VLC       # Job name
#SBATCH --output=slurm_out/debug_attention.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --gres=shard:10              # Number of GPUs
#SBATCH --partition=partition-1
#SBATCH --time=72:00:00

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate roboclip





python debug_attention_model.py \
    --seed 42 \
    --text_string "pressing button from side" \
    --model_base_path "/scr/yusenluo/RoboCLIP/visualization/clip_liv_models/RegressionRandom_liv_subtract_before_heads_4" \
    --transform_model_path "model_74.pt"

