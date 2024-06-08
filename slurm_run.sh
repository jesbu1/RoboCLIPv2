#!/bin/bash
#SBATCH --job-name=example_job     # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/example_job.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/example_job.err    # Error file
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --gres=gpu:1             # Number of GPUs



# Run the srun command
srun python metaworld_envs.py --env-type "sparse_learnt" --env-id "door-close-v2-goal-hidden" --dir-add "/scr/jzhang96/metaworld_log/" --text-string "robot close the door"