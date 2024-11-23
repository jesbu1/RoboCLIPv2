#!/bin/bash
#SBATCH --job-name=2       # Job name
#SBATCH --output=2.out   # Output file
#SBATCH --error=2.err    # Error file
#SBATCH --ntasks=2                      # Number of tasks (processes)
#SBATCH --gres=shard:15                    # Number of GPUs                
#SBATCH --cpus-per-task=5               # Number of CPU cores per task






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

# python category_training.py --model_name liv --sample_neg &
# python category_training.py --model_name liv --subtract --sample_neg &
# python category_training.py --model_name liv --pca --sample_neg &
# python category_training.py --model_name liv --pca --subtract --sample_neg &
# python category_training.py --model_name liv --pca --pca_var 0.95 --sample_neg &
# python category_training.py --model_name liv --pca --pca_var 0.95 --subtract --sample_neg &


# python regression_video_pca.py --pca --sample_neg --subtract_before &
# python regression_video_pca.py --pca  --subtract_before &

python regression_video_pca.py --sample_neg --subtract_before &
python regression_video_pca.py  --subtract_before &

# python regression_video_pca.py --pca --sample_neg --attention_heads 1 &
# python regression_video_pca.py --pca --attention_heads 1 &

# python regression_video_pca.py --sample_neg --attention_heads 1 &
# python regression_video_pca.py --attention_heads 1 &

wait
