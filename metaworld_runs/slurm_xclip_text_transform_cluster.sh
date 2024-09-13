#!/bin/bash
#SBATCH --job-name=PCAseed3242      # Job name
#SBATCH --output=PCAseed3242.out   # Output file
#SBATCH --error=PCAseed3242.err    # Error file
#SBATCH --ntasks=12                      # Number of tasks (processes)
#SBATCH --time=72:00:00                  # Time limit hrs:min:sec
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --partition=debug
#SBATCH --exclude=ink-gary,ink-lucy,ink-ron,lime-mint,allegro-chopin,dill-sage
#SBATCH --gres=gpu:a6000:1

source ~/miniconda3/etc/profile.d/conda.sh

conda info --envs
conda deactivate
conda deactivate


conda activate roboclip
which python



seeds=(32 42)

for seed in "${seeds[@]}"; do
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --env_id 'sweep-v2-goal-hidden' --baseline --text_string 'sweeping bin' --exp_name_end "v1baseline" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --env_id 'push-v2-goal-hidden' --baseline --text_string 'pushing bin' --exp_name_end "v1baseline" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --env_id 'handle-pull-side-v2-goal-hidden' --baseline --text_string 'pulling handle' --exp_name_end "v1baseline" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'coffee-pull-v2-goal-hidden' --text_string 'pulling cup' --exp_name_end "10_12PCAOrg" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'door-unlock-v2-goal-hidden' --text_string 'unlocking door' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'door-open-v2-goal-hidden' --text_string 'opening door' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'drawer-open-v2-goal-hidden' --text_string 'opening drawer' --exp_name_end "10_12PCAOrg" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'handle-pull-side-v2-goal-hidden' --text_string 'pullinig handle' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'faucet-open-v2-goal-hidden' --text_string 'opening faucet' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'stick-push-v2-goal-hidden' --text_string 'pushing stick' --exp_name_end "10_12PCAOrg" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'window-close-v2-goal-hidden' --text_string 'closing window' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'sweep-v2-goal-hidden' --text_string 'sweeping bin' --exp_name_end "10_12PCAOrg" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "/home/yusenluo/RoboCLIPv2/visualization/jzhang96_losses/triplet_loss_subset_6_42RE_NormVLM_PCAtriplet" --transform_model_path "org_model.pth" --env_id 'push-v2-goal-hidden' --text_string 'pushing bin' --exp_name_end "10_12PCAOrg" & 
done

wait