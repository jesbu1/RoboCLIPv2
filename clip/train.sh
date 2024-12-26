#!/bin/bash
#SBATCH --job-name=4       # Job name
#SBATCH --output=4.out   # Output file
#SBATCH --error=4.err    # Error file
#SBATCH --ntasks=2                      # Number of tasks (processes)
#SBATCH --gres=shard:30                    # Number of GPUs                
#SBATCH --cpus-per-task=6               # Number of CPU cores per task




python regression_video_same_length_multi_head.py --sample_negative --random_shuffle --attention_heads 8 --cat_after &
python regression_video_same_length_multi_head.py --sample_negative --attention_heads 8 --cat_after &
python regression_video_same_length_multi_head.py --random_shuffle --attention_heads 8 --cat_after &
python regression_video_same_length_multi_head.py --cat_after --attention_heads 8 &
# wait


# python regression_video_pca.py --sample_neg  --attention_heads 4 &
# python regression_video_pca.py  --sample_neg --random_shuffle --attention_heads 4 &



# python regression_video_pca.py  --subtract_before &

# python regression_video_pca.py --pca --sample_neg --attention_heads 1 --subtract_before &
# python regression_video_pca.py --pca --attention_heads 1 --subtract_before &

python regression_video_pca.py --sample_neg --attention_heads 8 &
python regression_video_pca.py --attention_heads 8 &

wait
