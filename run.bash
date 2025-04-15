# Status:
python test_scripts/test_iql.py koch=multi_task algorithm=wsrl_iql reward=rewind_two_cam general_training.action_chunk_size=60 environment.robot_disabled=True offline_training.offline_training_steps=150000 online_training.total_time_steps=0 logging.video_freq=0 general_training.awr_advantage_temp=1.0

# Status: Running
python test_scripts/test_iql.py koch=multi_task algorithm=wsrl_iql reward=rewind_two_cam general_training.action_chunk_size=60 environment.robot_disabled=True offline_training.offline_training_steps=150000 online_training.total_time_steps=0 logging.video_freq=0 general_training.awr_advantage_temp=0.5


# Status: Running
python test_scripts/test_iql.py koch=multi_task algorithm=wsrl_iql reward=rewind_two_cam general_training.action_chunk_size=60 environment.robot_disabled=True offline_training.offline_training_steps=150000 online_training.total_time_steps=0 logging.video_freq=0 general_training.awr_advantage_temp=2.5
