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
import imageio
import wandb
import io
import random
import torch.nn.functional as F
import h5py
import json


def eval_policys(args, env, policy):
    succ_count = 0
    total_count = 0
    for seed in range(400, 500):
        eval_env = env(args)
        obs = eval_env.reset_seed(seed)
        img_buffer = []
        for i in range(500):
            action, _states = policy.predict(obs)
            obs, rewards, dones, info = eval_env.step(action)
            if info['success']:
                succ_count += 1
                break
        total_count += 1
    return succ_count/total_count