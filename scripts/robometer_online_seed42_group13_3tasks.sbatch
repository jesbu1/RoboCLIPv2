#!/usr/bin/env bash
# Seed42 Robometer online jobs bound to the dedicated group13 server.
# Balanced 3-task split:
#   0 sweep-into-v2 baseline
#   1 sweep-into-v2 diff_gamma099
#   2 door-lock-v2 baseline

#SBATCH --job-name=robo_on_s42_g13_3
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PROJECT_DIR
export GROUP_ID=13
export INFO_FILE="${PROJECT_DIR}/logs/robometer_server_seed42_group13_info.txt"

bash "${PROJECT_DIR}/scripts/robometer_online_seed42_group_generic_3tasks.sbatch"
