# Train reward function
## Metaworld
CUDA_VISIBLE_DEVICES=1 python train_roll_progress_only_ema.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --openx_data --rewind_ratio 0.0 --epochs 25 --extra_data_type metaworld --openx_data --positional_encoding --end_rewind_ratio 0.1 --view all --data_type new

## Koch
python train_roll_progress_only_ema.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --openx_data --rewind_ratio 0.8 --epochs 100 --extra_data_type real_world --openx_data --positional_encoding --end_rewind_ratio 0.1 --view all --data_type new

## check the reward model save dir in PC
## Then need to load the pre-trained model in policy training
