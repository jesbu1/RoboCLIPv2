#!/bin/bash
#SBATCH --job-name=xclip_bonus       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/sac_xclip_text_baselines.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/sac_xclip_text_baselines.err    # Error file
#SBATCH --ntasks=2                      # Number of tasks (processes)
#SBATCH --gres=shard:20                        # Number of GPUs
#SBATCH --cpus-per-task=9               # Number of CPU cores per task




seeds=(42 32 5 0 1)
for seed in "${seeds[@]}"; do
    python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --succ_end --env_id 'button-press-v2-goal-hidden' --time_reward 100 --succ_bonus 100 --text_string 'pressing button' &  
    python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --succ_end --env_id 'button-press-topdown-v2-goal-hidden' --time_reward 100 --succ_bonus 100 --text_string 'pressing button' &  

    wait
done