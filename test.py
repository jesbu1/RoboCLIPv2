from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor
import numpy as np
import random
import json
import argparse
import os
# from stable_baselines.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from gym.wrappers.time_limit import TimeLimit
import cv2
import imageio
from tqdm import tqdm
import PIL.Image



def save_array_video(array, filename, fps=30):
    # save as mp4


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (array.shape[2], array.shape[1]))
    for i in range(array.shape[0]):
        out.write(array[i, :, :, ::-1])
    out.release()

def save_array_gif(array, filename, fps=30):
    imageio.mimsave(filename, array, fps=fps)



'''
METAWORLD DEMO GENERATION:
Plan:
1. We choose tasks from MT50, we can choose number of tasks for demo generation
2. We choose number of demos per task
3. The rest of the task we use for testing
4. Testing is given a task demo, we try to generate the reward function

Method:
Random shuffle the MT50 tasks, and save in a json file. We choose the first k task for demo generation, and the rest for testing.

Policy Training:
1. Load the task order
2. Load the tasks
3. Train the policy
4. Save the policy
'''


def main(args):
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # generate 50 random index
    random_indices = np.random.permutation(50)

    # get the tasks
    if args.goal_hidden:
        MT50_tasks = list(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys())
    else:
        MT50_tasks = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())

    # get the tasks
    # check file task_order exist
    task_name = "task_order_" + str(args.seed) + ".json"

    video_save_path = os.path.join(args.output_dir, "videos")
    policy_save_path = os.path.join(args.output_dir, "policies")

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    if not os.path.exists(policy_save_path):
        os.makedirs(policy_save_path)
    if os.path.exists(task_name):
        with open(task_name, 'r') as f:
            random_indices = json.load(f)
            random_indices = np.array(random_indices)
    else:
        with open(task_name, 'w') as f:
            # change to list
            random_indices = random_indices.tolist()
            json.dump(random_indices, f)

    # get the tasks
    # demo_tasks = random_indices[:args.demo_tasks]
    # test_tasks = random_indices[args.demo_tasks:]
    # demo_tasks = [MT50_tasks[i] for i in demo_tasks]
    # test_tasks = [MT50_tasks[i] for i in test_tasks]
    MT50_tasks = [MT50_tasks[i] for i in random_indices]
    model = None
    for task_name in MT50_tasks:
        if args.goal_hidden:
            task = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[task_name]
        else:
            task = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
        
        env = task(args.seed)
        env = TimeLimit(env, max_episode_steps=128)

        for i in range (5):
            obs = env.reset()
            img = env.render(mode='rgb_array')
            # save img to png
            img = PIL.Image.fromarray(img)
            img.save('meta_world_{0}_{1}.png'.format(task_name, i))
        import pdb ; pdb.set_trace()



if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_tasks", type=int, default=10)
    parser.add_argument("--demos_per_training_task", type=int, default=10)
    parser.add_argument("--demos_per_testing_task", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="/scr/jzhang96/metaworld_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal_hidden", action="store_true")
    parser.add_argument("--train_steps", type=int, default=100000)
    args = parser.parse_args()

    main(args)