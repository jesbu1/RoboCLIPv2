#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=slurm_out/roboclip_s3d_seed0.txt
#SBATCH --ntasks=1
#SBATCH --time=144:00:00
#SBATCH --partition=partition-1
#SBATCH --gres=gpu:1


# 激活 Conda 环境
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh  # 这里需要替换成你的实际 conda.sh 路径
conda activate roboclip      # 替换 myenv 为你的 Conda 环境名

# 执行 Python 脚本
python metaworld_envs_s3d_wandb.py --n_envs 1 --wandb --seed 0 --total_time-steps 200000 --env_id 'drawer-open-v2-goal-hidden' --text_string 'opening drawer'
#--filter True
# 如果需要，可以在这里添加更多的命令
