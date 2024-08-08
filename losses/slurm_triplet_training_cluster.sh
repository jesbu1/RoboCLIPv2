#!/bin/bash
#SBATCH --job-name=triplet_training      # Job name
#SBATCH --output=/home/jzhang96/metaworld_log/triplet_training.out   # Output file
#SBATCH --error=/home/jzhang96/metaworld_log/triplet_training.err    # Error file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=gpu:a6000:1                        # Number of GPUs
#SBATCH --cpus-per-task=18               # Number of CPU cores per task
#SBATCH --nodes=1
#SBTCH --nodelist=ink-titan
#SBATCH --time=48:00:00                  # Time limit hrs:min:sec



# #SBATCH --gres=gpu:1  # Number of GPUs

python s3d_text_triplet_loss_training.py --time_shuffle --time_shorten --norm --num_workers 16 --batch_size 64 --epochs 500
# python s3d_text_triplet_loss_training.py --time_shorten --norm --num_workers 17 --batch_size 64  


