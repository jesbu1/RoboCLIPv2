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



seeds=(42 32 5)

for seed in "${seeds[@]}"; do



    # python metaworld_envs_xclip_text_clean.py --n_envs 4 --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --model_base_path models/triplet_loss_subset_0_42_var1.0new_TimeShuffle_TimeShort_NormVLMtriplet --transform_model_path model_19999.pth --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "new_log_noPCA" & 
    # python metaworld_envs_xclip_text_clean.py --n_envs 4 --succ_bonus 200 --time_reward 100 --random_reset --seed $seed --model_base_path models/triplet_loss_subset_0_42_var1.0new_TimeShuffle_TimeShort_NormVLMtriplet --transform_model_path model_19999.pth --env_id 'handle-pull-side-v2-goal-hidden' --text_string 'pullinig handle' --exp_name_end "no_succ_end" & 

    # python metaworld_envs_xclip_text_clean.py --n_envs 4 --succ_bonus 200 --time_reward 100 --random_reset --seed $seed --model_base_path models/triplet_loss_subset_0_42_var1.0new_TimeShuffle_TimeShort_NormVLMtriplet --transform_model_path model_19999.pth --env_id 'window-close-v2-goal-hidden' --text_string 'closing window' --exp_name_end "no_succ_end" & 
    # python metaworld_envs_xclip_text_clean.py --n_envs 4 --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --model_base_path models/triplet_loss_subset_0_42_var1.0new_TimeShuffle_TimeShort_NormVLMtriplet --transform_model_path model_19999.pth --env_id 'door-open-v2-goal-hidden' --text_string 'opening door' --exp_name_end "new_log_noPCA" & 


    
    # python metaworld_envs_xclip_RMS.py --n_envs 4 --succ_bonus 50 --time_reward 1 --random_reset --seed $seed --baseline --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "RMS_BSL" &
    python metaworld_envs_xclip_RMS.py --n_envs 4 --succ_bonus 50 --time_reward 1 --random_reset --seed $seed --baseline --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "RMS_BSL" &


done

wait
