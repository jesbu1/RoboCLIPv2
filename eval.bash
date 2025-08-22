python test_scripts/eval.py koch=multi_task algorithm=rlpd_iql reward=vlc_two_cam general_training.action_chunk_size=60 evaluation.model_type=rlpd evaluation.n_episodes=10 evaluation.model_path="/home/abrar/projects/RoboCLIPv2/outputs/2025-05-13/14-20-55/logs/rlpd/last_offline.zip" environment.max_episode_steps=400 general_training.entropy_term=20 general_training.ensembling_param=0.1

python test_scripts/eval.py koch=multi_task algorithm=rlpd_iql reward=vlc_two_cam general_training.action_chunk_size=60 evaluation.model_type=rlpd evaluation.n_episodes=10 evaluation.model_path="/home/abrar/projects/RoboCLIPv2/outputs/2025-05-13/14-47-12/logs/rlpd.zip" environment.max_episode_steps=400 general_training.entropy_term=20 general_training.ensembling_param=0.1
# koch=multi_task algorithm=rlpd_iql reward=vlc_two_cam logging.video_freq=0 general_training.learning_rate=1e-4 offline_training.offline_training_steps=20000 offline_training.ckpt_path=/home/abrar/projects/RoboCLIPv2/outputs/2025-04-27/17-41-57/logs/rlpd/last_offline.zip general_training.entropy_term=10 general_training.ensembling_param=0.01

python test_scripts/eval.py koch=multi_task algorithm=rlpd_iql reward=vlc_two_cam general_training.action_chunk_size=60 evaluation.model_type=rlpd evaluation.n_episodes=10 evaluation.model_path="/home/abrar/projects/RoboCLIPv2/outputs/2025-05-13/14-20-55/logs/rlpd/last_offline.zip" environment.max_episode_steps=400 general_training.entropy_term=20 general_training.ensembling_param=0.1 #in distribution
"/home/abrar/projects/RoboCLIPv2/outputs/2025-05-13/18-39-58/logs/rlpd"


python test_scripts/eval.py koch=multi_task algorithm=rlpd_iql reward=vlc_two_cam general_training.action_chunk_size=60 evaluation.model_type=rlpd evaluation.n_episodes=10 evaluation.model_path="/home/abrar/projects/RoboCLIPv2/outputs/2025-05-14/21-06-59/logs/rlpd.zip" environment.max_episode_steps=400 general_training.entropy_term=10 general_training.ensembling_param=0.01 #in distribution(hard)


python test_scripts/test_iql.py koch=multi_task algorithm=rlpd_iql reward=vlc_two_cam logging.video_freq=0 general_training.learning_rate=1e-4 general_training.entropy_term=10.0 general_training.ensembling_param=0.01 offline_training.ckpt_path=/home/abrar/projects/RoboCLIPv2/outputs/2025-05-13/14-20-55/logs/rlpd/last_offline.zip
