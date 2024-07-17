#!/bin/bash
#SBATCH --job-name=sac_s3d_baselines       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/s3d_baselines.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/s3d_baselines.err    # Error file
#SBATCH --ntasks=12                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=4               # Number of CPU cores per task


PYTHON_SCRIPT_1="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term "auto" --time_penalty 0.1 --succ_bonus 100 --total_time_steps 500000" 
PYTHON_SCRIPT_2="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term 0.2 --time_penalty 0.1 --succ_bonus 100 --total_time_steps 500000"
PYTHON_SCRIPT_3="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term "auto" --time_penalty 0.1 --total_time_steps 500000"
PYTHON_SCRIPT_4="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term 0.2 --time_penalty 0.1 --total_time_steps 500000"
PYTHON_SCRIPT_5="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term "auto" --succ_bonus 100 --total_time_steps 500000"
PYTHON_SCRIPT_6="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term 0.2 --succ_bonus 100 --total_time_steps 500000"

PYTHON_SCRIPT_7="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term "auto" --time_penalty 0.1 --succ_bonus 500 --total_time_steps 500000" 
PYTHON_SCRIPT_8="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term 0.2 --time_penalty 0.1 --succ_bonus 500 --total_time_steps 500000"
PYTHON_SCRIPT_9="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term "auto" --time_penalty 0.1 --total_time_steps 500000"
PYTHON_SCRIPT_10="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term 0.2 --time_penalty 0.1 --total_time_steps 500000"
PYTHON_SCRIPT_11="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term "auto" --succ_bonus 500 --total_time_steps 500000"
PYTHON_SCRIPT_12="python metaworld_envs_s3d_sac_zeroshot.py --algo sac --n_envs 1 --wandb --eval_freq 512 --video_freq 10240 --succ_end --time --train_orcale --seed 42 --warm_up_runs 10 --norm_input --norm_output --time_100 --entropy_term 0.2 --succ_bonus 500 --total_time_steps 500000"



$PYTHON_SCRIPT_1 &
$PYTHON_SCRIPT_2 &
$PYTHON_SCRIPT_3 &
$PYTHON_SCRIPT_4 &
$PYTHON_SCRIPT_5 &
$PYTHON_SCRIPT_6 &
$PYTHON_SCRIPT_7 &
$PYTHON_SCRIPT_8 &
$PYTHON_SCRIPT_9 &
$PYTHON_SCRIPT_10 &
$PYTHON_SCRIPT_11 &
$PYTHON_SCRIPT_12 &

# Wait for all background jobs to finish
wait

