#!/bin/bash
#SBATCH --job-name=closing_window       # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/sac_s3d_text_transform.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/sac_s3d_text_transform.err    # Error file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=shard:30                        # Number of GPUs                
#SBATCH --cpus-per-task=9               # Number of CPU cores per task



# #SBATCH --gres=gpu:1  # Number of GPUs

python metaworld_envs_s3d_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --env_id 'door-open-v2-goal-hidden' --text_string 'closing door' --transform_model_path "triplet_loss_50_42_s3d_TimeShuffle_TimeShort_Normtriplet/1650.pth" --exp_name_end "Triplet_ours_Long" --total_time_steps 5000000 & 
python metaworld_envs_s3d_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --env_id 'door-open-v2-goal-hidden' --text_string 'closing door' --transform_model_path "triplet_loss_50_42_s3d_TimeShort_Normtriplet/1555.pth" --exp_name_end "Triplet_neg_shorten_Long" --total_time_steps 5000000 &
# python metaworld_envs_s3d_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --algo "sac" --env_id 'door-open-v2-goal-hidden' --text_string 'closing door' --transform_model_path "triplet_loss_50_42_s3d_TimeShuffle_TimeShort_Normtriplet/1650.pth" --exp_name_end "Triplet_ours" &&
# python metaworld_envs_s3d_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 0 --algo "sac" --env_id 'door-open-v2-goal-hidden' --text_string 'closing door' --transform_model_path "triplet_loss_50_42_s3d_TimeShuffle_TimeShort_Normtriplet/1650.pth" --exp_name_end "Triplet_ours" &&
# python metaworld_envs_s3d_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --time_reward 100 --entropy_term "auto" --random_reset --eval_freq 1280 --video_freq 5120 --seed 1 --algo "sac" --env_id 'door-open-v2-goal-hidden' --text_string 'closing door' --transform_model_path "triplet_loss_50_42_s3d_TimeShuffle_TimeShort_Normtriplet/1650.pth" --exp_name_end "Triplet_ours" &&


# Wait for all background jobs to finish
wait

