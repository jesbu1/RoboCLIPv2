#!/bin/bash
#SBATCH --job-name=4       # Job name
#SBATCH --output=4.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=shard:16                      # Number of GPUs                
#SBATCH --cpus-per-task=4               # Number of CPU cores per task

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate roboclip

python reward_model_decoder_5_demos_training.py --sample_neg --attention_heads 4 --reverse_video --normalize_embedding --positional_encoding

wait

