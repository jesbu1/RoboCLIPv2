import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt

# read csv file

#S3D training files
s3d_dir = "/scr/jzhang96/metaworld/door-close-v2-goal-hidden_sparse_learnt/scr/jzhang96/metaworld_log/s3d"
training_files_0 = os.path.join(s3d_dir, "0.monitor.csv")
training_files_1 = os.path.join(s3d_dir, "1.monitor.csv")
training_files_2 = os.path.join(s3d_dir, "2.monitor.csv")
training_files_3 = os.path.join(s3d_dir, "3.monitor.csv")

eval_files_0 = os.path.join(s3d_dir, "10.monitor.csv")
eval_files_1 = os.path.join(s3d_dir, "11.monitor.csv")
eval_files_2 = os.path.join(s3d_dir, "12.monitor.csv")
eval_files_3 = os.path.join(s3d_dir, "13.monitor.csv")

xclip_dir = "/scr/jzhang96/metaworld/door-close-v2-goal-hidden_sparse_learntmetaworld_log/xclip_render_fix"
xclip_train_0 = os.path.join(xclip_dir, "0.monitor.csv")
xclip_train_1 = os.path.join(xclip_dir, "1.monitor.csv")
xclip_train_2 = os.path.join(xclip_dir, "2.monitor.csv")
xclip_train_3 = os.path.join(xclip_dir, "3.monitor.csv")

xclip_eval_0 = os.path.join(xclip_dir, "10.monitor.csv")
xclip_eval_1 = os.path.join(xclip_dir, "11.monitor.csv")
xclip_eval_2 = os.path.join(xclip_dir, "12.monitor.csv")
xclip_eval_3 = os.path.join(xclip_dir, "13.monitor.csv")



# read csv files
s3d_train_0 = pd.read_csv(training_files_0, skiprows=1)["r"].to_numpy()
s3d_train_1 = pd.read_csv(training_files_1, skiprows=1)["r"].to_numpy()
s3d_train_2 = pd.read_csv(training_files_2, skiprows=1)["r"].to_numpy()
s3d_train_3 = pd.read_csv(training_files_3, skiprows=1)["r"].to_numpy()

s3d_eval_0 = pd.read_csv(eval_files_0, skiprows=1)["r"].to_numpy()
s3d_eval_1 = pd.read_csv(eval_files_1, skiprows=1)["r"].to_numpy()
s3d_eval_2 = pd.read_csv(eval_files_2, skiprows=1)["r"].to_numpy()
s3d_eval_3 = pd.read_csv(eval_files_3, skiprows=1)["r"].to_numpy()

xclip_train_0 = pd.read_csv(xclip_train_0, skiprows=1)["r"].to_numpy()
xclip_train_1 = pd.read_csv(xclip_train_1, skiprows=1)["r"].to_numpy()
xclip_train_2 = pd.read_csv(xclip_train_2, skiprows=1)["r"].to_numpy()
xclip_train_3 = pd.read_csv(xclip_train_3, skiprows=1)["r"].to_numpy()

xclip_eval_0 = pd.read_csv(xclip_eval_0, skiprows=1)["r"].to_numpy()
xclip_eval_1 = pd.read_csv(xclip_eval_1, skiprows=1)["r"].to_numpy()
xclip_eval_2 = pd.read_csv(xclip_eval_2, skiprows=1)["r"].to_numpy()
xclip_eval_3 = pd.read_csv(xclip_eval_3, skiprows=1)["r"].to_numpy()

# convert to array
min_episodes = min(len(s3d_train_0), len(s3d_train_1), len(s3d_train_2), len(s3d_train_3), 
                     len(xclip_train_0), len(xclip_train_1), len(xclip_train_2), len(xclip_train_3))
print("min_episodes: ", min_episodes)

data_group_1 = np.stack([s3d_train_0[:min_episodes], s3d_train_1[:min_episodes], s3d_train_2[:min_episodes], s3d_train_3[:min_episodes]])
data_group_2 = np.stack([xclip_train_0[:min_episodes], xclip_train_1[:min_episodes], xclip_train_2[:min_episodes], xclip_train_3[:min_episodes]])

import pdb ; pdb.set_trace()
df_group1 = pd.DataFrame(data_group_1.T, columns=["s3d_train_0", "s3d_train_1", "s3d_train_2", "s3d_train_3"])
df_group2 = pd.DataFrame(data_group_2.T, columns=["xclip_train_0", "xclip_train_1", "xclip_train_2", "xclip_train_3"])

df_group1["time_step"] = df_group1.index * 128
df_group2["time_step"] = df_group2.index * 128

df_group1["group"] = "s3d"
df_group2["group"] = "xclip"


df_group1 = df_group1.melt(id_vars=["time_step", "group"], var_name="s3d", value_name="reward")
df_group2 = df_group2.melt(id_vars=["time_step", "group"], var_name="xclip", value_name="reward")



# import pdb ; pdb.set_trace()

df_summary_group1 = df_group1.groupby('time_step')['reward'].agg(['mean', 'std']).reset_index()
df_summary_group2 = df_group2.groupby('time_step')['reward'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df_summary_group1, x='time_step', y='mean', label='s3d')
sns.lineplot(data=df_summary_group2, x='time_step', y='mean', label='xclip')

plt.fill_between(df_summary_group1['time_step'], df_summary_group1['mean'] - df_summary_group1['std'], df_summary_group1['mean'] + df_summary_group1['std'], alpha=0.3)
plt.fill_between(df_summary_group2['time_step'], df_summary_group2['mean'] - df_summary_group2['std'], df_summary_group2['mean'] + df_summary_group2['std'], alpha=0.3)

plt.xlabel('Time Step')
plt.ylabel('Mean Reward')
plt.title('Training Reward Comparison')
plt.legend()

# save png
plt.savefig('training_reward_comparison.png')


# only for group 2 training
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_summary_group2, x='time_step', y='mean', label='xclip')
plt.fill_between(df_summary_group2['time_step'], df_summary_group2['mean'] - df_summary_group2['std'], df_summary_group2['mean'] + df_summary_group2['std'], alpha=0.3)

plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Training Reward Comparison')
plt.legend()

# save png
plt.savefig('training_reward_comparison-2.png')

data_group_3 = np.stack([s3d_eval_0[:min_episodes], s3d_eval_1[:min_episodes], s3d_eval_2[:min_episodes], s3d_eval_3[:min_episodes]])
data_group_4 = np.stack([xclip_eval_0[:min_episodes], xclip_eval_1[:min_episodes], xclip_eval_2[:min_episodes], xclip_eval_3[:min_episodes]])

df_group3 = pd.DataFrame(data_group_3.T, columns=["s3d_eval_0", "s3d_eval_1", "s3d_eval_2", "s3d_eval_3"])
df_group4 = pd.DataFrame(data_group_4.T, columns=["xclip_eval_0", "xclip_eval_1", "xclip_eval_2", "xclip_eval_3"])

df_group3["time_step"] = df_group3.index * 2560
df_group4["time_step"] = df_group4.index * 2560

df_group3["group"] = "s3d"
df_group4["group"] = "xclip"

df_group3 = df_group3.melt(id_vars=["time_step", "group"], var_name="s3d", value_name="reward")
df_group4 = df_group4.melt(id_vars=["time_step", "group"], var_name="xclip", value_name="reward")

df_summary_group3 = df_group3.groupby('time_step')['reward'].agg(['mean', 'std']).reset_index()
df_summary_group4 = df_group4.groupby('time_step')['reward'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df_summary_group3, x='time_step', y='mean', label='s3d')
sns.lineplot(data=df_summary_group4, x='time_step', y='mean', label='xclip')

plt.fill_between(df_summary_group3['time_step'], df_summary_group3['mean'] - df_summary_group3['std'], df_summary_group3['mean'] + df_summary_group3['std'], alpha=0.3)
plt.fill_between(df_summary_group4['time_step'], df_summary_group4['mean'] - df_summary_group4['std'], df_summary_group4['mean'] + df_summary_group4['std'], alpha=0.3)

plt.xlabel('Time Step')
plt.ylabel('Mean Reward')
plt.title('Evaluation Reward Comparison')

plt.legend()

# save png
plt.savefig('evaluation_reward_comparison.png')

# only plot s3d evaluation plot
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_summary_group3, x='time_step', y='mean', label='s3d')
plt.fill_between(df_summary_group3['time_step'], df_summary_group3['mean'] - df_summary_group3['std'], df_summary_group3['mean'] + df_summary_group3['std'], alpha=0.3)

plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Evaluation Reward s3d')


# save png
plt.savefig('evaluation_reward_comparison_s3d.png')