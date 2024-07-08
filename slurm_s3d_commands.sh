#!/bin/bash
#SBATCH --job-name=s3d_runs     # Job name
#SBATCH --output=/home/jzhang96/slurm_logs/slurm.out   # Output file
#SBATCH --error=/home/jzhang96/slurm_logs/slurm.err    # Error file
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --gres=gpu:2080:2             # Number of GPUs
#SBATCH --nodelist=ink-gary
#SBATCH --time=20:00:00


# Run the srun command
# origin roboclip
env_id="window-close-v2-goal-hidden"
text_string="closing window"
seed=42
target_gif_path="metaworld_generate_gifs"

# srun python metaworld_envs_s3d_fix_norm.py --target_gif_path "$target_gif_path" --n_envs 1 --wandb --succ_end --train_orcale --time_100 --env_id "$env_id" --text_string "$text_string" --seed "$seed" &
# no threshold
# srun python metaworld_envs_s3d_fix_norm.py --target_gif_path "$target_gif_path" --n_envs 1 --wandb --succ_end --train_orcale --warm_up_runs 10 --succ_end --norm_input --norm_output --time_100 --env_id "$env_id" --text_string "$text_string" --seed "$seed" &
# threshold 
# srun python metaworld_envs_s3d_fix_norm.py --target_gif_path "$target_gif_path" --n_envs 1 --wandb --succ_end --train_orcale --warm_up_runs 10 --succ_end --norm_input --norm_output --time_100 --threshold_reward --env_id "$env_id" --text_string "$text_string" --seed "$seed" &
# threshold + project
# srun python metaworld_envs_s3d_fix_norm.py --target_gif_path "$target_gif_path" --n_envs 1 --wandb --succ_end --train_orcale --warm_up_runs 10 --succ_end --project_reward --norm_input --norm_output --time_100 --threshold_reward --env_id "$env_id" --text_string "$text_string" --seed "$seed" &
 
# wait

# echo "All commands have finished."


# srun --gres=gpu:2080:2 --nodelist ink-lisa --time 1 nvidia-smi
