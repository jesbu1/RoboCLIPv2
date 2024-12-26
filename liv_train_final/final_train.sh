#!/bin/bash
#SBATCH --job-name=4       # Job name
#SBATCH --output=4.out   # Output file
#SBATCH --error=4.err    # Error file
#SBATCH --ntasks=4                      # Number of tasks (processes)
#SBATCH --gres=shard:30                    # Number of GPUs                
#SBATCH --cpus-per-task=4               # Number of CPU cores per task

python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --normalize_embedding & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video --normalize_embedding & 
wait
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subtract_before & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --subtract_before & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --normalize_embedding --subtract_before & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video --normalize_embedding --subtract_before & 
wait
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress & 
wait
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subtract_before --catagorical_progress_bins 5 --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --subtract_before --catagorical_progress_bins 5 --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --normalize_embedding --subtract_before --catagorical_progress_bins 5 --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video --normalize_embedding --subtract_before --catagorical_progress_bins 5 --catagorical_progress & 
wait
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subsample_video & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --subsample_video & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --normalize_embedding --subsample_video & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video --normalize_embedding --subsample_video & 
wait
python reward_model_training.py --sample_neg  --attention_heads 4  & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video  & 
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video &  
python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video  & 
wait
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding 

# python reward_model_training.py --sample_neg  --attention_heads 4 --subtract_before
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subtract_before

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress 
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress --subtract_before
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --subtract_before




# python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding  --subsample_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --subtract_before --subsample_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subtract_before --subsample_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress  --subsample_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --subsample_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress --subtract_before --subsample_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --subtract_before --subsample_video



# # reverse video

# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --reverse_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --subtract_before --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subtract_before --reverse_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress  --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --reverse_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress --subtract_before --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --subtract_before --reverse_video




# python reward_model_training.py --sample_neg  --attention_heads 4 --subsample_video --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding  --subsample_video --reverse_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --subtract_before --subsample_video --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --subtract_before --subsample_video --reverse_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress  --subsample_video --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --subsample_video --reverse_video

# python reward_model_training.py --sample_neg  --attention_heads 4 --catagorical_progress_bins 5 --catagorical_progress --subtract_before --subsample_video --reverse_video
# python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress_bins 5 --catagorical_progress --subtract_before --subsample_video --reverse_video


