
# Two Step Training
# 1. No PE
# 2. PE only add on the first frame
# 3. PE add on fifst and last frame
CUDA_VISIBLE_DEVICES=1 python train_roll_abrar.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --binary_threshold 0.50 --rewind_ratio 0.5 --epochs 30 --progress_loss_weight 2 --extra_data_type metaworld --openx_data &&
CUDA_VISIBLE_DEVICES=1 python train_roll_abrar.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --binary_threshold 0.50 --rewind_ratio 0.5 --epochs 30 --progress_loss_weight 2 --extra_data_type metaworld --openx_data --positional_encoding &&
CUDA_VISIBLE_DEVICES=1 python train_roll_abrar.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --binary_threshold 0.50 --rewind_ratio 0.5 --epochs 30 --progress_loss_weight 2 --extra_data_type metaworld --openx_data --positional_encoding --last_frame_pe &&
wait

# One Step Training
# 1. No PE
# 2. PE only add on the first frame
# 3. PE add on fifst and last frame
CUDA_VISIBLE_DEVICES=1 python train_roll_progress_only.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --openx_data --rewind_ratio 0.5 --epochs 20 --progress_loss_weight 2 --extra_data_type metaworld --openx_data &&
CUDA_VISIBLE_DEVICES=1 python train_roll_progress_only.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --openx_data --rewind_ratio 0.5 --epochs 20 --progress_loss_weight 2 --extra_data_type metaworld --openx_data --positional_encoding &&
CUDA_VISIBLE_DEVICES=1 python train_roll_progress_only.py --rewind --subsample_video --max_length 16 --two_step_training --cosine_scheduler --clip_grad --progress_loss --extra_data_ratio 0.20 --text_embedding_model minilm --worker 1 --openx_data --rewind_ratio 0.5 --epochs 20 --progress_loss_weight 2 --extra_data_type metaworld --openx_data --positional_encoding --last_frame_pe &&
wait
