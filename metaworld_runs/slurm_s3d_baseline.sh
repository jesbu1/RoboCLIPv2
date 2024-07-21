#!/bin/bash
#SBATCH --job-name=ppo_s3d_baselines       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/s3d_transform.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/s3d_transform.err    # Error file
#SBATCH --ntasks=3                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=9               # Number of CPU cores per task


PYTHON_SCRIPT_1="python metaworld_envs_s3d_baseline.py --n_envs 8 --wandb --time --train_orcale --norm_input --norm_output --time_100 --entropy_term 0.5 --random_reset --eval_freq 1280 --video_freq 5120 --seed 42" 
PYTHON_SCRIPT_2="python metaworld_envs_s3d_baseline.py --n_envs 8 --wandb --time --train_orcale --norm_input --norm_output --time_100 --entropy_term 0.5 --random_reset --eval_freq 1280 --video_freq 5120 --seed 5" 
PYTHON_SCRIPT_3="python metaworld_envs_s3d_baseline.py --n_envs 8 --wandb --time --train_orcale --norm_input --norm_output --time_100 --entropy_term 0.5 --random_reset --eval_freq 1280 --video_freq 5120 --seed 32" 



$PYTHON_SCRIPT_1 &
$PYTHON_SCRIPT_2 &
$PYTHON_SCRIPT_3 &


# Wait for all background jobs to finish
wait

