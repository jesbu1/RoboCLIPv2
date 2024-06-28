#!/bin/bash
#SBATCH --job-name=example_job     # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/example_job_xclip_render_fix_text_closing.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/example_job_xclip_render_fix_text_closing.err    # Error file
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --gres=gpu:1             # Number of GPUs



# Run the srun command
srun python metaworld_envs_xclip_wandb_fix_reset.py --env_type "sparse_learnt" --env_id "window-open-v2-goal-hidden" --dir_add "/scr/jzhang96/metaworld_log/" --text_string "opening window" --n_envs 1 --wandb --seed 42  
# srun python metaworld_envs.py --env-type "sparse_learnt" --env-id "drawer-close-v2-goal-hidden" --dir-add "/scr/jzhang96/metaworld_log/" --text-string "robot closing green drawer" --n-envs 2 &
# wait
