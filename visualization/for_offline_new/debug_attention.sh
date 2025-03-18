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




python metaworld_envs_attention_new.py \
    --text_string "closing window" \
    --model_base_path "/scr/yusenluo/RoboCLIP/visualization/for_offline_new/roboclip_v2_models_final/RegressionRandom_liv_sample_neg_subtract_after_heads_4_sample_neg_reverse_video_norm" \
    --transform_model_path "model_149.pth"

