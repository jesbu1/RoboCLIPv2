import h5py

# h5_path = 'data/h5_buffers/orig/metaworld_traj_15_demos.h5'
h5_path = "data/h5_buffers/updated_trajs/metaworld_traj_15_demos_dense.h5"
# h5_path = 'data/h5_buffers/orig/metaworld_traj_15_demos.h5'
h5_file = h5py.File(h5_path, "r")

print(h5_file.keys())
breakpoint()

# print action statistics for each dimension (N, 4)
print(h5_file["action"].shape)
for i in range(4):
    print(
        f"Action {i} {h5_file['action'][:, i].mean()} +/- {h5_file['action'][:, i].std()}"
    )
    print("Min/max", h5_file["action"][:, i].min(), h5_file["action"][:, i].max())
    print()

# draw a histogram and for each action in one plot
import matplotlib.pyplot as plt

plt.subplot(2, 2, 1)
plt.hist(h5_file["action"][:, 0])
plt.title("Action 0")

plt.subplot(2, 2, 2)
plt.hist(h5_file["action"][:, 1])
plt.title("Action 1")

plt.subplot(2, 2, 3)
plt.hist(h5_file["action"][:, 2])
plt.title("Action 2")

plt.subplot(2, 2, 4)
plt.hist(h5_file["action"][:, 3])
plt.title("Action 3")

plt.savefig("action_histogram.png")


# Let's also print the observation space and the min/max of its values
print(h5_file["state"].shape)

for i in range(39):
    print(
        f"Observation {i} {h5_file['state'][:, i].mean()} +/- {h5_file['state'][:, i].std()}"
    )
    print("Min/max", h5_file["state"][:, i].min(), h5_file["state"][:, i].max())
    print()


# breakpoint()

# import torch
# import numpy as np
# from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
# from envs.metaworld_envs.metaworld import create_wrapped_env


# lang_feat = torch.zeros(512)
# env_id = 'window-open-v2-goal-hidden'


# envs = DummyVecEnv([create_wrapped_env(env_id,  language_features=lang_feat, success_bonus=True, use_simulator_reward=True)])
# envs.reset()
# all_actions = []
# all_states = []
# for i in range(500):
#     # Sample random actions from the action space
#     action_space = envs.action_space
#     actions = [action_space.sample()]
#     # Take a step in the environment
#     obs, rewards, dones, info = envs.step(actions)
#     all_states.append(obs)

#     all_actions.append(actions[0])

# all_actions = np.array(all_actions)
# all_states = np.array(all_states)

# print(all_actions.shape)
# for i in range(4):
#     print(f"Action {i} {all_actions[:, i].mean()} +/- {all_actions[:, i].std()}")
#     print("Min/max", all_actions[:, i].min(), all_actions[:, i].max())
#     print()

# print(all_states.shape)
# all_states = all_states.reshape(-1, all_states.shape[-1])
# for i in range(39):
#     print(f"Observation {i} {all_states[:, i].mean()} +/- {all_states[:, i].std()}")
#     print("Min/max", all_states[:, i].min(), all_states[:, i].max())
#     print()

# print h5py strings
h5_strings = h5_file["string"]

strings_list = []
for string in h5_strings:
    strings_list.append(string.decode("utf-8"))

# Count how many times each item in the list appears
string_counts = {}
for string in strings_list:
    if string in string_counts:
        string_counts[string] += 1
    else:
        string_counts[string] = 1

print(string_counts)

import pdb

pdb.set_trace()
