#!/bin/bash
#SBATCH --job-name=hard_task_collection
#SBATCH --output=srun_output/output_%j.txt
#SBATCH --ntasks=1                   # Single task for this job
#SBATCH --cpus-per-task=19            # Allocate enough CPUs for all parallel tasks
#SBATCH --gres=gpu:1                 # Allocate 1 GPU for the job

# Load necessary modules
# module load python/3.8

# Run the script 5 times with different seeds on the same GPU
# finish tasks: 0,1,2,3,4,5,6,7,13,14,15,16,48,49,    8,9,10,11,12,18,19,  22,23,24,25,26,35,36,37,38, 29,30,31,32,33,34,41,42,43,44,45 
# have enough tasks:
# for i in {8,9,10}
# for i in {8,9,10}
# for i in {8,9,10,11,12,18,19}
# hard task: 0,1,2,3,5,9,12,19,27,28,29,30,34,39,41,42,45,47
# easier task: 17, 32,33,40,43
for i in {0,1,2,3,5,9,12,19,27,28,29,30,34,39,41,42,45,47}
do
    SEED=$i
    python -m meta_world_sac_training --task_id $SEED --train_steps 2560000 &
done

# Wait for all background jobs to finish
wait
