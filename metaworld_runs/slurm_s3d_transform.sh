#!/bin/bash
#SBATCH --job-name=ppo_s3d_baselines       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/s3d_transform.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/s3d_transform.err    # Error file
#SBATCH --ntasks=6                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=8               # Number of CPU cores per task


PYTHON_SCRIPT_1="python metaworld_envs_s3d_transform.py --n_envs 8 --wandb --train_orcale --norm_input --time_reward 100 --entropy_term auto --random_reset --eval_freq 1280 --video_freq 5120 --algo sac --norm_output --seed 32 "
PYTHON_SCRIPT_2="python metaworld_envs_s3d_transform.py --n_envs 8 --wandb --train_orcale --norm_input --time_reward 100 --entropy_term auto --random_reset --eval_freq 1280 --video_freq 5120 --algo sac --norm_output --seed 5 "
PYTHON_SCRIPT_3="python metaworld_envs_s3d_transform.py --n_envs 8 --wandb --train_orcale --norm_input --time_reward 100 --entropy_term auto --random_reset --eval_freq 1280 --video_freq 5120 --algo sac --norm_output --seed 42 "

PYTHON_SCRIPT_1="python metaworld_envs_s3d_transform.py --n_envs 8 --wandb --train_orcale --norm_input --time_reward 100 --entropy_term 0.5 --random_reset --eval_freq 1280 --video_freq 5120 --algo ppo --norm_output --seed 32 "
PYTHON_SCRIPT_2="python metaworld_envs_s3d_transform.py --n_envs 8 --wandb --train_orcale --norm_input --time_reward 100 --entropy_term 0.5 --random_reset --eval_freq 1280 --video_freq 5120 --algo ppo --norm_output --seed 5 "
PYTHON_SCRIPT_3="python metaworld_envs_s3d_transform.py --n_envs 8 --wandb --train_orcale --norm_input --time_reward 100 --entropy_term 0.5 --random_reset --eval_freq 1280 --video_freq 5120 --algo ppo --norm_output --seed 42 "



$PYTHON_SCRIPT_1 &
$PYTHON_SCRIPT_2 &
$PYTHON_SCRIPT_3 &
$PYTHON_SCRIPT_4 &
$PYTHON_SCRIPT_5 &
$PYTHON_SCRIPT_6 &


# Wait for all background jobs to finish
wait

