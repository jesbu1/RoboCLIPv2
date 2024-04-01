#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=python_job_out.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
SBATCH --partition=partition-1

# 激活 Conda 环境
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh  # 这里需要替换成你的实际 conda.sh 路径
conda activate roboclip      # 替换 myenv 为你的 Conda 环境名

# 执行 Python 脚本
python /scr/yusenluo/RoboCLIP/visualization/task1.py  # 替换为你的 Python 脚本路径

# 如果需要，可以在这里添加更多的命令
