#!/bin/bash
#SBATCH --job-name=4       # Job name
#SBATCH --output=4.out   # Output file
#SBATCH --error=4.err    # Error file
#SBATCH --ntasks=12                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                   # Number of GPUs                
#SBATCH --cpus-per-task=4               # Number of CPU cores per task


python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --two_step_training &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --two_step_training --catagorical_progress &

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --two_step_training &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --two_step_training --catagorical_progress &

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --two_step_training &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --two_step_training --catagorical_progress &

wait

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --two_step_training &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --two_step_training --catagorical_progress &

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --catagorical_progress --subsample_video --max_length 16 & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --two_step_training --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --fully_reverse_data --two_step_training --catagorical_progress --subsample_video --max_length 16 &

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --catagorical_progress --subsample_video --max_length 16 & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --two_step_training --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --normalize_embedding --two_step_training --catagorical_progress --subsample_video --max_length 16 &

wait

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --catagorical_progress --subsample_video --max_length 16 & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --two_step_training --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --fully_reverse_data --two_step_training --catagorical_progress --subsample_video --max_length 16 &

python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --catagorical_progress --subsample_video --max_length 16 & 
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --two_step_training --subsample_video --max_length 16 &
python reward_model_training.py --sample_neg  --attention_heads 4 --reverse_video --two_step_training --catagorical_progress --subsample_video --max_length 16 &

python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --catagorical_progress & 
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding &
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --two_step_training &
python reward_model_training.py --sample_neg  --attention_heads 4 --normalize_embedding --two_step_training --catagorical_progress &

wait

