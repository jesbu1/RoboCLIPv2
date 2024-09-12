#!/bin/bash
#SBATCH --job-name=seed05       # Job name
#SBATCH --output=seed05.out   # Output file
#SBATCH --error=seed05.err    # Error file
#SBATCH --ntasks=24                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=1               # Number of CPU cores per task


# seeds=(42 32 5 0 1)
# seeds=(0 1)

# for seed in "${seeds[@]}"; do
    # pca ours
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'door-unlock-v2-goal-hidden' --text_string 'unlocking door' --exp_name_end "Set0" & 

    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'coffee-pull-v2-goal-hidden' --text_string 'pulling cup' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'door-open-v2-goal-hidden' --text_string 'opening door' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'drawer-open-v2-goal-hidden' --text_string 'opening drawer' --exp_name_end "Set0" & 

    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'handle-pull-side-v2-goal-hidden' --text_string 'pulling handle' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'faucet-open-v2-goal-hidden' --text_string 'opening faucet' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'stick-push-v2-goal-hidden' --text_string 'pushing stick' --exp_name_end "Set0" & 

    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'window-close-v2-goal-hidden' --text_string 'closing window' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'sweep-v2-goal-hidden' --text_string 'sweeping bin' --exp_name_end "Set0" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'push-v2-goal-hidden' --text_string 'pushing bin' --exp_name_end "Set0" & 
#     wait
# done

# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 5 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# wait
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 0 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 1 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 1 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# wait
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 5 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# wait
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_end --time_reward 100 --succ_bonus 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 0 --algo "sac" --transform_base_path "/scr/jzhang96/triplet_text_loss_models" --transform_model_path "triplet_loss_50_42_xclip_TimeShuffle_TimeShort_Norm_NormVLM_RandomNoisetriplet_Aug/880.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "OursLong" & 
# wait

# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 300 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 32 --algo "sac" --transform_base_path "models/triplet_loss_subset_6_42_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'push-v2-goal-hidden' --text_string 'pushing bin' --exp_name_end "Set6_300" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 300 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 5 --algo "sac" --transform_base_path "models/triplet_loss_subset_6_42_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'push-v2-goal-hidden' --text_string 'pushing bin' --exp_name_end "Set6_300" & 
# python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 300 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed 42 --algo "sac" --transform_base_path "models/triplet_loss_subset_6_42_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'push-v2-goal-hidden' --text_string 'pressing bin' --exp_name_end "Set6_300" & 
# wait


seeds=(0 5)

for seed in "${seeds[@]}"; do
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --env_id 'sweep-v2-goal-hidden' --baseline --text_string 'sweeping bin' --exp_name_end "v1baseline" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --env_id 'push-v2-goal-hidden' --baseline --text_string 'pushing bin' --exp_name_end "v1baseline" & 
    # python metaworld_envs_xclip_text_transform.py --n_envs 8 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 100 --succ_end --time_reward 100 --random_reset --eval_freq 1280 --video_freq 5120 --seed $seed --algo "sac" --env_id 'handle-pull-side-v2-goal-hidden' --baseline --text_string 'pulling handle' --exp_name_end "v1baseline" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'button-press-topdown-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'button-press-v2-goal-hidden' --text_string 'pressing button' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'coffee-pull-v2-goal-hidden' --text_string 'pulling cup' --exp_name_end "fixSAC" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'door-unlock-v2-goal-hidden' --text_string 'unlocking door' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'door-open-v2-goal-hidden' --text_string 'opening door' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'drawer-open-v2-goal-hidden' --text_string 'opening drawer' --exp_name_end "fixSAC" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'handle-pull-side-v2-goal-hidden' --text_string 'pullinig handle' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'faucet-open-v2-goal-hidden' --text_string 'opening faucet' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'stick-push-v2-goal-hidden' --text_string 'pushing stick' --exp_name_end "fixSAC" & 

    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'window-close-v2-goal-hidden' --text_string 'closing window' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'sweep-v2-goal-hidden' --text_string 'sweeping bin' --exp_name_end "fixSAC" & 
    python metaworld_envs_xclip_text_transform.py --n_envs 1 --wandb --time --norm_input --norm_output --entropy_term "auto" --succ_bonus 200 --succ_end --time_reward 100 --random_reset --seed $seed --algo "sac" --transform_base_path "models/triplet_loss_subset_0_42RE_TimeShuffle_TimeShort_NormVLMtriplet" --transform_model_path "model_9999.pth" --env_id 'push-v2-goal-hidden' --text_string 'pushing bin' --exp_name_end "fixSAC" & 
done

wait
