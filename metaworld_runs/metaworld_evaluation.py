
from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image, ImageSequence
import torch as th
import numpy as np
from PIL import Image, ImageSequence
import cv2
import gif2numpy
import PIL
import os
import seaborn as sns
import matplotlib.pylab as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from typing import Any, Dict

import gym
from gym.spaces import Box
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import os
from stable_baselines3.common.monitor import Monitor
from memory_profiler import profile
import argparse
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

# from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt

from gym.wrappers import RecordVideo
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import io
import random
import torch.nn.functional as F
import h5py
import json
from s3dg import S3D
from transformers import AutoTokenizer, AutoModel, AutoProcessor 



'''
1. Regular RoboCLIP v1
2. RoboCLIP v1 with single seed
3. RoboCLIP v1 with multiple seeds
4. RoboCLIP v1 with same video demo
5. RoboCLIP v1 with pca
6. RoboCLIP v1 with pca and single seed
7. RoboCLIP v1 with pca and multiple seeds
8. RoboCLIP v1 with pca and same video demo
fix gpu, forward xclip with gpu Done

'''

id_task = json.load(open("../id_task.json", "r"))




def adjust_frames_xclip(frames, target_frame_count = 32, processor = None):
    """
    Ensures same numbers of frames(32). returns a numpy array of shape (target_frame_count, 224, 224, 3)
    """
    frames = np.array(frames)
    frame_count = frames.shape[0]
    #print(f"frames number{frame_count}")
    # frames = th.from_numpy(frames)

    if len(frames) > target_frame_count:
        index = np.linspace(0, len(frames)-1, target_frame_count, dtype=int)
        frames = frames[index]
    elif len(frames) < target_frame_count:
        last_frame = frames[-1]
        last_frame = np.expand_dims(last_frame, axis=0)
        for _ in range(target_frame_count - len(frames)):
            frames = np.concatenate([frames, last_frame])
    frames = frames[:,240-112:240+112,320-112:320+112,:]
    # frames = frames[None, :,:,:,:]
    frames = processor(videos=list(frames), return_tensors="pt")
    frames = frames["pixel_values"]
    return frames





def parse_entropy_term(value):
    try:
        return float(value)
    except ValueError:
        return value


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac'])
    parser.add_argument('--text_string', type=str, default='closing door')
    parser.add_argument('--dir_add', type=str, default='')
    parser.add_argument('--env_id', type=str, default='button-press-v2-goal-hidden')
    parser.add_argument('--total_time_steps', type=int, default=1000000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=1280)
    parser.add_argument('--video_freq', type=int, default=5120)
    parser.add_argument('--succ_end', action="store_true")
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--time', action="store_false")
    parser.add_argument('--frame_num', type=int, default=32)
    parser.add_argument('--train_orcale', action="store_true") # load latent from h5 file
    parser.add_argument('--warm_up_runs', type=int, default=0)
    parser.add_argument('--project_reward', action="store_true")
    parser.add_argument('--norm_output', action="store_true")
    parser.add_argument('--time_100', action="store_true")
    parser.add_argument('--threshold_reward', action="store_true")
    parser.add_argument('--time_penalty', type=float, default=0.0)
    parser.add_argument('--xclip_model', type=str, default='microsoft/xclip-base-patch16-zero-shot')
    parser.add_argument('--frame_length', type=int, default=32)

    args = parser.parse_args()
    return args




class MetaworldDense(Env):
    def __init__(self, args, seed=0):
        super(MetaworldDense, self)
        self.door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[args.env_id]
        self.args = args
        self.rank = seed
        self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
        self.env = TimeLimit(self.baseEnv, max_episode_steps=128)
        self.time = args.time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        
        self.counter = 0
        self.counter_total = 0
        self.gif_buffer = []

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
        
    
    def render(self, camera_name="topview"):
        frame = self.env.render()

        return frame


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.counter += 1
        self.counter_total += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])


        if args.succ_end:
            if info['success']:
                done = True

        return obs, reward, done, info
        
    def reset(self):
        self.counter = 0

        # if self.args.random_reset:
        #     self.rank = random.randint(400, 500)
        #     self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
        #     # env = door_open_goal_hidden_cls(seed=rank)
        #     self.env = TimeLimit(self.baseEnv, max_episode_steps=128)
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])




def make_env(args):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():

        env = MetaworldDense(args)
        env = Monitor(env, os.path.join(log_dir, str(args.seed)))
        return env
    return _init





def main():
    global args
    global log_dir
    args = get_args()

    # set seed
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    experiment_name = "s3d_evaluation_" + args.algo + "_" + args.env_id 
    # experiment_name = "s3d_sac_baseline_" + args.env_id + "_" + args.algo + "_" + str(args.seed)
    run_group = "s3d_evaluation"

    # if args.wandb:
    #     run = wandb.init(
    #         entity=WANDB_ENTITY_NAME,
    #         project=WANDB_PROJECT_NAME,
    #         group=run_group,
    #         config=args,
    #         name=experiment_name,
    #         monitor_gym=True,
    #         sync_tensorboard=True,
    #     )



    # log_dir = f"/scr/jzhang96/logs/baseline_logs/{experiment_name}"

    # log_dir = f"/scr/jzhang96/logs/baseline_logs/{experiment_name}"

    # log_dir = f"/home/jzhang96/logs/baseline_logs/{experiment_name}"


    # Evaluate the agent
    seed_num = ["5", "32", "42", "0", "1"]
    # seed_num = ["42"]
    total_sr = 0
    for seed_str in seed_num:
        model = SAC.load(f"/home/jzhang96/logs/baseline_logs/xclip_textTRANS_sac_button-press-v2-goal-hidden_XReward100.0_SuccEndMILNCELong_{seed_str}NEW/best_model.zip")
    # model = PPO.load(f"/scr/jzhang96/logs/baseline_logs/s3d_baseline_ppo_door-close-v2-goal-hidden_Oracle_NormIn_XReward100.0_NoTime_RandReset_Entropyauto_5/best_model.zip")
        succ_count = 0
        total_count = 0
        for seed in range(400, 500):
            eval_env = MetaworldDense(args, seed=seed)
            obs = eval_env.reset()
            img_buffer = []
            for i in range(128):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = eval_env.step(action)
                import pdb; pdb.set_trace()
                if seed == 400:
                    img = eval_env.render()
                    img_buffer.append(img)
                if info['success']:
                    succ_count += 1
                    break
            if seed == 400:
                imageio.mimsave(f"{seed_str}_seed.gif", img_buffer)
            total_count += 1

        print(f"seed {seed_str}, success rate {succ_count/total_count}")
        total_sr += succ_count/total_count
    print(f"average success rate {total_sr/5}")






if __name__ == '__main__':
    main()