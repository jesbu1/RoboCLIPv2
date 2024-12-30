#!/bin/bash
#SBATCH --job-name=VLC       # Job name
#SBATCH --output=slurm_out/v2.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --gres=shard:10              # Number of GPUs
#SBATCH --partition=partition-1
#SBATCH --time=72:00:00

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate roboclip


seeds=(0)

# env_ids=(
#     "button-press-topdown-v2"
#     "button-press-v2"
#     "coffee-pull-v2"
#     "door-open-v2"
#     "door-unlock-v2"
#     "drawer-open-v2"
#     "handle-pull-side-v2"
#     "faucet-open-v2"
#     "stick-push-v2"
#     "window-close-v2"
#     "sweep-v2"
#     "push-v2"
# )

env_ids=(
  "faucet-open-v2"
)


declare -A env_text_map=(
    ["button-press-topdown-v2"]="pressing button from top"
    ["button-press-wall-v2"]="pressing button from side"
    ["button-press-v2"]="pressing button from side"
    ["coffee-pull-v2"]="pulling cup"
    ["coffee-push-v2"]="pushing coffee cup"
    ["door-open-v2"]="opening door"
    ["door-unlock-v2"]="unlocking door"
    ["drawer-open-v2"]="opening drawer"
    ["drawer-close-v2"]="closing drawer"
    ["handle-pull-side-v2"]="pulling handle"
    ["handle-press-side-v2"]="pressing handle from side"
    ["faucet-open-v2"]="opening faucet"
    ["faucet-close-v2"]="closing faucet"
    ["stick-push-v2"]="pushing stick"
    ["stick-pull-v2"]="pulling stick"
    ["window-close-v2"]="closing window"
    ["window-open-v2"]="opening window"
    ["sweep-v2"]="sweeping block"
    ["push-v2"]="pushing block"
)

max_jobs=1
current_jobs=0

for seed in "${seeds[@]}"; do
  for env_id in "${env_ids[@]}"; do
    text_string="${env_text_map[$env_id]}"

    python metaworld_envs_attention_new.py \
      --n_envs 4 \
      --succ_bonus 10 \
      --succ_end \
      --time_reward 1 \
      --random_reset \
      --seed $seed \
      --env_id "$env_id-goal-hidden" \
      --text_string "$text_string" \
      --ep_length 128 \
      --model_base_path "/scr/yusenluo/RoboCLIP/visualization/for_offline_new/roboclip_v2_models_final/RegressionRandom_liv_sample_neg_subtract_after_heads_4_sample_neg_reverse_video_norm" \
      --transform_model_path "model_149.pth" \
      --exp_name_end "V2" &

    ((current_jobs+=1))

    if [ "$current_jobs" -ge "$max_jobs" ]; then
      wait -n
      ((current_jobs-=1))
    fi
  done
done