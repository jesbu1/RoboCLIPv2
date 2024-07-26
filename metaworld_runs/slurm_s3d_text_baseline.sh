#!/bin/bash
#SBATCH --job-name=sac_s3d_text_baselines       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/sac_s3d_text_baselines.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/sac_s3d_text_baselines.err    # Error file
#SBATCH --ntasks=3                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=9               # Number of CPU cores per task


PYTHON_SCRIPT_1="python metaworld_envs_s3d_text_baseline.py --n_envs 8 --wandb --time --norm_input --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --norm_output --algo "sac" "
PYTHON_SCRIPT_2="python metaworld_envs_s3d_text_baseline.py --n_envs 8 --wandb --time --norm_input --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 5 --norm_output --algo "sac" " 
PYTHON_SCRIPT_3="python metaworld_envs_s3d_text_baseline.py --n_envs 8 --wandb --time --norm_input --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --norm_output --algo "sac" " 



$PYTHON_SCRIPT_1 &
$PYTHON_SCRIPT_2 &
$PYTHON_SCRIPT_3 &


# Wait for all background jobs to finish
wait

