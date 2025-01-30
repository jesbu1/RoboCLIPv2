#!/bin/bash
#SBATCH --job-name=4       # Job name
#SBATCH --output=eval_liv.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=shard:16                      # Number of GPUs                
#SBATCH --cpus-per-task=1               # Number of CPU cores per task

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate roboclip

python eval_rewind.py


