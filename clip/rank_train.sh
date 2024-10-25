#!/bin/bash
#SBATCH --job-name=rank_train      # Job name
#SBATCH --output=/scr/jzhang96/metaworld_log/rank_training.out   # Output file
#SBATCH --error=/scr/jzhang96/metaworld_log/rank_training.err    # Error file
#SBATCH --ntasks=12                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --nodes=1



python rank_loss_training.py --sample_other_task --batch_size 64 --hard_margin_loss soft_margin --last_state_loss --epochs 5000 &
python rank_loss_training.py --sample_other_task --batch_size 64 --hard_margin_loss hard_margin --last_state_loss --epochs 5000 &
python rank_loss_training.py --sample_other_task --batch_size 64 --hard_margin_loss cross_entropy --last_state_loss --epochs 5000 & 
python rank_loss_training.py --batch_size 64 --hard_margin_loss cross_entropy --last_state_loss --epochs 5000 &
python rank_loss_training.py --batch_size 64 --hard_margin_loss hard_margin --last_state_loss --epochs 5000 &
python rank_loss_training.py --batch_size 64 --hard_margin_loss soft_margin --last_state_loss --epochs 5000 &

python rank_loss_training.py --sample_other_task --batch_size 64 --hard_margin_loss soft_margin --last_state_loss --epochs 5000 --model_name liv &
python rank_loss_training.py --sample_other_task --batch_size 64 --hard_margin_loss hard_margin --last_state_loss --epochs 5000 --model_name liv &
python rank_loss_training.py --sample_other_task --batch_size 64 --hard_margin_loss cross_entropy --last_state_loss --epochs 5000 --model_name liv &
python rank_loss_training.py --batch_size 64 --hard_margin_loss cross_entropy --last_state_loss --epochs 5000 --model_name liv &
python rank_loss_training.py --batch_size 64 --hard_margin_loss hard_margin --last_state_loss --epochs 5000 --model_name liv &
python rank_loss_training.py --batch_size 64 --hard_margin_loss soft_margin --last_state_loss --epochs 5000 --model_name liv &
wait




