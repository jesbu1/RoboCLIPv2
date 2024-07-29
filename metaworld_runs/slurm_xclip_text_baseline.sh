#!/bin/bash
#SBATCH --job-name=xclip_runs       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/sac_xclip_text_baselines.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/sac_xclip_text_baselines.err    # Error file
#SBATCH --ntasks=2                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=10               # Number of CPU cores per task




python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --env_id 'door-close-v2-goal-hidden' --text_string 'closing door' &
python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 5 --algo "sac" --env_id 'door-close-v2-goal-hidden' --text_string 'closing door' &
# python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --algo "sac" --env_id 'door-close-v2-goal-hidden' --text_string 'closing door' &
# python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 0 --algo "sac" --env_id 'door-close-v2-goal-hidden' --text_string 'closing door' & 
# python metaworld_envs_xclip_text_baseline.py --n_envs 8 --wandb --time --norm_input --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 1 --algo "sac" --env_id 'door-close-v2-goal-hidden' --text_string 'closing door' & 


# Wait for all background jobs to finish
wait

