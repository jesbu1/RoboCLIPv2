#!/bin/bash
#SBATCH --job-name=sac_xclip_text_trans      # Job name
#SBATCH --output=sac_xclip_text_baselines7.out   # Output file
#SBATCH --error=sac_xclip_text_baselines7.err    # Error file
#SBATCH --ntasks=3                      # Number of tasks (processes)
#SBATCH --time=48:00:00                  # Time limit hrs:min:sec
#SBATCH --cpus-per-task=10               # Number of CPU cores per task
#SBATCH --partition=debug
#SBATCH --exclude=ink-gary,ink-lucy,ink-ron,lime-mint,ink-mia,ink-noah,allegro-chopin,glamor-ruby
#SBATCH --gres=gpu:a6000:1

source ~/miniconda3/etc/profile.d/conda.sh

conda info --envs
conda deactivate
conda deactivate


conda activate roboclip
which python



# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --transform_base_path "/home/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Normtriplet/345.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursRERE" &
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --random_reset --eval_freq 1280 --video_freq 5120 --seed 5 --algo "sac" --transform_base_path "/home/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Normtriplet/345.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursRERE" &
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --algo "sac" --transform_base_path "/home/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Normtriplet/345.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursRERE" &
python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --random_reset --eval_freq 1280 --video_freq 5120 --seed 0 --algo "sac" --transform_base_path "/home/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Normtriplet/345.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursRERE" & 
python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --random_reset --eval_freq 1280 --video_freq 5120 --seed 1 --algo "sac" --transform_base_path "/home/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Normtriplet/345.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursRERE" & 


# Wait for all background jobs to finish
wait