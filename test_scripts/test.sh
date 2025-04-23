#!/bin/bash
#SBATCH --job-name=dense_bsl       # Job name
#SBATCH --output=test_scripts/slurm_out/dense_bsl_coffee_push_5.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=32               # Number of CPU cores per task
#SBATCH --exclude=ink-gary,ink-lucy,ink-ron,ink-lisa,ink-ellie,lime-mint,allegro-chopin,ink-noah,ink-mia,dill-sage
#SBATCH --mem=128G                       # Total memory for all tasks
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec

source /home/yusenluo/anaconda3/etc/profile.d/conda.sh  # 这里需要替换成你的实际 conda.sh 路径
conda activate roboclip      # 替换 myenv 为你的 Conda 环境名

python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=sac reward=dense
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=sac reward=rewind > test_scripts/slurm_out/rewind_debug.txt
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=wsrl_iql reward=rewind > test_scripts/slurm_out/rewind_wsrl_iql_debug.txt
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=sac reward=gvl
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=sac reward=liv > test_scripts/slurm_out/liv_debug.txt
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=sac reward=sparse > test_scripts/slurm_out/sparse_debug.txt
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=ppo reward=sparse > test_scripts/slurm_out/ppo_sparse_debug.txt

python test_scripts/test_iql.py metaworld=off_on_15 algorithm=wsrl_iql reward=rewind > test_scripts/slurm_out/rewind_wsrl_iql_debug.txt
# bash /home/yusenluo/rewind/RoboCLIPv2/docker/run_singularity_slurm_script.sh ' python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=sac reward=dense'

python test_scripts/test_iql.py metaworld=off_on_15 algorithm=rlpd_iql reward=rewind
done