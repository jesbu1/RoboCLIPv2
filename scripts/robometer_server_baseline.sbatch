#!/usr/bin/env bash
#SBATCH --job-name=robo_srv_bl
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/robometer_server_baseline_%j.out
#SBATCH --error=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/robometer_server_baseline_%j.err

set -e

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project2/biyik_1165/haobaizh/rewind_robometer_valuemodel/conda_envs/robometer

SERVER_HOST=$(hostname)
SERVER_PORT=8000
INFO_FILE=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/robometer_server_baseline_info.txt
echo "http://${SERVER_HOST}:${SERVER_PORT}" > "$INFO_FILE"
echo "Server at: http://${SERVER_HOST}:${SERVER_PORT}"

cd /project2/biyik_1165/haobaizh/rewind_robometer_valuemodel
python -u robometer_server.py
