#!/bin/bash
#SBATCH --job-name=1       # Job name
#SBATCH --output=1.out   # Output file
#SBATCH --error=1.err    # Error file
#SBATCH --ntasks=6                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                    # Number of GPUs                
#SBATCH --cpus-per-task=4               # Number of CPU cores per task






# python regression_training.py --model_name liv &
# python regression_training.py --model_name liv --subtract &
# python regression_training.py --model_name liv --pca &
# python regression_training.py --model_name liv --pca --subtract &
# python regression_training.py --model_name liv --pca --pca_var 0.95 &
# python regression_training.py --model_name liv --pca --pca_var 0.95 --subtract &


# python category_training.py --model_name liv &
# python category_training.py --model_name liv --subtract &
# python category_training.py --model_name liv --pca &
# python category_training.py --model_name liv --pca --subtract &
# python category_training.py --model_name liv --pca --pca_var 0.95 &
# python category_training.py --model_name liv --pca --pca_var 0.95 --subtract &

python category_training.py --model_name liv --sample_neg &
python category_training.py --model_name liv --subtract --sample_neg &
python category_training.py --model_name liv --pca --sample_neg &
python category_training.py --model_name liv --pca --subtract --sample_neg &
python category_training.py --model_name liv --pca --pca_var 0.95 --sample_neg &
python category_training.py --model_name liv --pca --pca_var 0.95 --subtract --sample_neg &


wait
