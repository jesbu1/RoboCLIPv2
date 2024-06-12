from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image, ImageSequence
import torch as th
from s3dg import S3D
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
from stable_baselines3.common.callbacks import EvalCallback

import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoProcessor 
from gym.wrappers import RecordVideo
import imageio

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text-string', type=str, default='robot opening sliding door')
    parser.add_argument('--dir-add', type=str, default='')
    parser.add_argument('--env-id', type=str, default='AssaultBullet-v0')
    parser.add_argument('--env-type', type=str, default='AssaultBullet-v0')
    parser.add_argument('--total-time-steps', type=int, default=5000000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    # parser.add_argument('--xclip_model', type=str, default='microsoft/xclip-base-patch16-zero-shot')


    args = parser.parse_args()
    return args
class MetaworldSparse(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
        super(MetaworldSparse,self)
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        # self.net = S3D('s3d_dict.npy', 512)
        # # Load the model weights
        # self.net.load_state_dict(th.load('s3d_howto100m.pth'))

        # load xclip model
        model_name = "microsoft/xclip-base-patch16-zero-shot"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.net = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)


        # Evaluation mode
        self.net = self.net.eval()
        self.target_embedding = None
        if text_string:
            # text_output = self.net.text_module([text_string])
            # self.target_embedding = text_output['text_embedding']
            text_tokens = self.tokenizer([text_string], return_tensors="pt")
            self.target_embedding = self.net.get_text_features(**text_tokens)


        if video_path:
            frames = readGif(video_path)
            
            if human:
                frames = self.preprocess_human_demo(frames)
            else:
                frames = self.preprocess_metaworld_xclip(frames)
            if frames.shape[1]>3:
                frames = frames[:,:3]
            video = th.from_numpy(frames)
            video_output = self.net(video.float())
            self.target_embedding = video_output['video_embedding']
        assert self.target_embedding is not None

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames

    def preprocess_metaworld_xclip(self, frames, shorten=True):
        center = 240, 320
        # h, w = (250, 250)
        h, w = (224, 224)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
        # frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        # frames = frames[None, :,:,:,:] 
        length = len(frames)
        if shorten:
            frames = [frames[i][y:y+h, x:x+w] for i in range (0, length, 4)]
        else:
            frames = [frame[y:y+h, x:x+w] for frame in frames]

        # path = "test_imgs"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # for i, frame in enumerate(frames):
        #     cv2.imwrite(f"{path}/frame_{i}.png", frame)
        # import imageio
        # output_file = 'output.gif'

        # Save the frames as a GIF

        frames = self.processor(videos=frames, return_tensors="pt")
        frames = frames["pixel_values"]
        # frames = frames["pixel_values"].cpu().detach().numpy()


        # frames = frames.transpose(0, 4, 1, 2, 3)


        # frames = frames/255
        return frames


    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld_xclip(self.past_observations)
            
            
            # video = th.from_numpy(frames)
            video_embedding = self.net.get_video_features(frames)

            # video_embedding = video_output['video_embedding']
            similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())

            reward = similarity_matrix.detach().numpy()[0][0]
            return obs, reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])


class MetaworldDense(Env):
    def __init__(self, env_id, time=False, rank=0):
        super(MetaworldDense,self)
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        
        self.counter = 0
        self.counter_total = 0
        self.gif_buffer = []
        self.rank = rank

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
        
    
    def render(self, camera_name="topview"):
        # camera_name="topview"
        frame = self.env.render()
        # self.gif_buffer.append(frame)

        return frame


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # self.past_observations.append(self.env.render())
        self.counter += 1
        self.counter_total += 1
        t = self.counter/128
        self.gif_buffer.append(self.env.render())
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
            # save gif

            if t - 1 == 0:
                if self.counter_total % 1280 == 0:
                    path = "/scr/jzhang96/metaworld_gifs/xclip/"+"door_opening_"+str(self.rank)
                    # path = "metaworld_gifs/xclip/"+"door_opening_"+str(self.rank)

                    if not os.path.exists(path):
                        os.makedirs(path)
                    file_name = f'{path}/output_{self.counter_total}.gif'
                    # Save the frames as a GIF
                    imageio.mimsave(file_name, self.gif_buffer, duration=0.1)
                self.gif_buffer = []

        

            
        return obs, reward, done, info
        
    def reset(self):
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])




def make_env(env_type, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if env_type == "sparse_learnt":
            env = MetaworldSparse(env_id=env_id, text_string="robot closing green drawer", time=True, rank=rank)
            # env = MetaworldSparse(env_id=env_id, video_path="./gifs/human_opening_door.gif", time=True, rank=rank, human=True)
        
        elif env_type == "sparse_original":
            env = KitchenEnvSparseOriginalReward(time=True)
        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init



def make_env_render(env_type, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if env_type == "sparse_learnt":
            env = MetaworldSparse(env_id=env_id, text_string="robot closing green drawer", time=True, rank=rank)
            # env = MetaworldSparse(env_id=env_id, video_path="./gifs/human_opening_door.gif", time=True, rank=rank, human=True)
        
        elif env_type == "sparse_original":
            env = KitchenEnvSparseOriginalReward(time=True)
        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank)
        log_dir_video =  os.path.join(log_dir, str(rank))
        env = Monitor(env, log_dir_video)
        env = RecordVideo(env, log_dir_video, episode_trigger=lambda episode_id: True, name_prefix="eval")
        env = DummyVecEnv([lambda: env])
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init








def main():
    global args
    global log_dir
    args = get_args()
    env_id = "button-press-v2-goal-hidden"
    log_dir = f"/scr/jzhang96/metaworld/{args.env_id}_{args.env_type}{args.dir_add}xclip_render_fix"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, i) for i in range(args.n_envs)])

    if not args.pretrained:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, i) for i in range(10, 10+args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=1000,
                                 deterministic=True, render=False)


    # eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                              log_path=log_dir, eval_freq=300,
    #                              deterministic=True, render=False)

    # eval_callback = VideoEvalCallback(eval_env=eval_env,
    #                                     video_folder=log_dir,
    #                                     best_model_save_path=log_dir,
    #                                     log_path=log_dir,
    #                                     eval_freq=300,
    #                                     deterministic=True,
    #                                     render=True)

    model.learn(total_timesteps=int(args.total_time_steps), callback=eval_callback)
    model.save(f"{log_dir}/trained")



if __name__ == '__main__':
    main()
