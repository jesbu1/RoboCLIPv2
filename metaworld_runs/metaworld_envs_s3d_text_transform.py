
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




def adjust_frames_s3d(frames, target_frame_count = 32):
    """
    Ensures same numbers of frames(32). 
    """
    frames = np.array(frames)
    #print(frames.shape)
    if len(frames) > 32:
        index = np.linspace(0, len(frames)-1, 32, dtype=int)
        frames = frames[index]
    elif len(frames) < 32:
        last_frame = frames[-1]
        last_frame = np.expand_dims(last_frame, axis=0)
        #print(last_frame.shape)
        for _ in range(32 - len(frames)):
            frames = np.concatenate([frames, last_frame])
    frames = frames[:,240-125:240+125,320-125:320+125,:]
    frames = frames[None, :,:,:,:]
    frames = frames.transpose(0, 4, 1, 2, 3)

    return frames

def parse_entropy_term(value):
    try:
        return float(value)
    except ValueError:
        return value

class SingleLayerMLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

class SimpleWeightVector(th.nn.Module):
    def __init__(self, dim):
        super(SimpleWeightVector, self).__init__()
        self.weight = th.nn.Parameter(th.randn(dim, dim))

    def forward(self, x):
        # x: (batch_size, dim)
        # self.weight: (1, dim)
        # output: (batch_size, dim)
        return x * self.weight


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
    parser.add_argument('--algo', type=str, default='sac', choices=['ppo', 'sac'])
    parser.add_argument('--text_string', type=str, default='opening door')
    parser.add_argument('--dir_add', type=str, default='')
    parser.add_argument('--env_id', type=str, default='window-open-v2-goal-hidden')
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
    parser.add_argument('--random_reset', action="store_true")
    # parser.add_argument('--target_gif_path', type=str, default="/scr/jzhang96/metaworld_generate_gifs/")
    parser.add_argument('--target_gif_path', type=str, default="/home/jzhang96/RoboCLIPv2/metaworld_generate_gifs/")
    parser.add_argument('--time', action="store_false")
    parser.add_argument('--frame_num', type=int, default=32)
    parser.add_argument('--train_orcale', action="store_true") # load latent from h5 file
    parser.add_argument('--warm_up_runs', type=int, default=0)
    parser.add_argument('--project_reward', action="store_true")
    parser.add_argument('--norm_input', action="store_true")
    parser.add_argument('--norm_output', action="store_true")
    parser.add_argument('--time_reward', type=float, default=1.0)
    parser.add_argument('--threshold_reward', action="store_true")
    parser.add_argument('--entropy_term', type=parse_entropy_term, default="auto")
    parser.add_argument('--time_penalty', type=float, default=0.0)
    parser.add_argument('--succ_bonus', type=float, default=0.0)
    parser.add_argument('--transform_base_path', type=str, default="/scr/jzhang96/triplet_text_loss_models")
    parser.add_argument('--transform_model_path', type=str, default="triplet_loss_50_42_s3d_Normtriplet/1650.pth")
    parser.add_argument("--exp_name_end", type=str, default="triplet_hard_neg")

    # parser.add_argument('--xclip_model', type=str, default='microsoft/xclip-base-patch16-zero-shot')


    args = parser.parse_args()
    return args

class MetaworldSparse(Env):
    # def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
    def __init__(self, args):
        super(MetaworldSparse,self)
        self.args = args
        self.door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[args.env_id]
        self.rank = args.seed
        self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
        self.env = TimeLimit(self.baseEnv, max_episode_steps=128)
        self.time = args.time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []

        with th.no_grad():
            self.net = S3D('../s3d_dict.npy', 512)
            self.net.load_state_dict(th.load('../s3d_howto100m.pth'))
            # self.net = self.net.eval().cuda()
            self.net = self.net.eval()
            self.target_embedding = None

            # self.transform_model = SingleLayerMLP(512, 512, normalize=True).cuda()
            # self.transform_model.load_state_dict(th.load("/scr/jzhang96/triplet_loss_models/triplet_loss_45_42_s3d_0.0_TimeShuffle_TimeShort_Norm/49.pth"))
            self.transform_model = SingleLayerMLP(512, 512, normalize=True)
            transform_model_path = os.path.join(args.transform_base_path, args.transform_model_path)
            self.transform_model.load_state_dict(th.load(transform_model_path))
            self.transform_model = self.transform_model.eval().cuda()


            if args.text_string:
                for _ in range (3):
                    print("text_string", args.text_string)
                text_string = args.text_string
                text_output = self.net.text_module([text_string])
                self.target_embedding = text_output['text_embedding'].cuda()
                self.net = self.net.eval().cuda()
                if args.norm_output:
                    self.target_embedding = normalize_embeddings(self.target_embedding, return_tensor=True).float()


            self.max_sim = None
            if args.warm_up_runs > 0:
                for _ in range(args.warm_up_runs):
                    embedding = self.warm_up_run()

                    if args.norm_output:
                        embedding = normalize_embeddings(embedding, return_tensor=True).float()
                    embedding = self.transform_model(embedding)
                    embedding = normalize_embeddings(embedding, return_tensor=True).float()

                    sim = th.matmul(self.target_embedding, embedding.t())
                    if self.args.time_reward != 1.0:
                        sim = sim * self.args.time_reward
                    if self.max_sim is None:
                        sim_reward = sim.detach().cpu().numpy()[0][0]
                        self.max_sim = sim_reward
                    else:
                        sim_reward = sim.detach().cpu().numpy()[0][0]
                        if sim_reward > self.max_sim:
                            self.max_sim = sim_reward
                    print("sim_reward", sim_reward, self.max_sim)
                print("max_sim", self.max_sim)

                # only input text or demo videos for now

            self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    


    def warm_up_run(self):
        self.env.reset()
        images = []
        frame_number = random.randint(32, 128)

        for _ in range(frame_number):
            action = self.env.action_space.sample()
            _, _, _, _ = self.env.step(action)
            images.append(self.env.render()[:,:,:3])

        with th.no_grad():
            frames = adjust_frames_s3d(images)
            if self.args.norm_input:
                frames = frames/255
            frames = th.from_numpy(frames).float().cuda()
            frames = self.net(frames)
            frames = frames['video_embedding']

        return frames
    
    def render(self):
        frame = self.env.render()
        return frame


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        
        if args.succ_end:
            if info['success']:
                done = True

        if done:
            with th.no_grad():
                frames = adjust_frames_s3d(self.past_observations)
                frames = th.from_numpy(frames).float().cuda()
                if self.args.norm_input:
                    frames = frames/255
                video_embedding = self.net(frames)['video_embedding']
                # used to test the model with norm embeddings
                if self.args.norm_output:
                    video_embedding = normalize_embeddings(video_embedding, return_tensor=True).float()
                    self.target_embedding = normalize_embeddings(self.target_embedding, return_tensor=True).float()
                video_embedding = self.transform_model(video_embedding)
                video_embedding = normalize_embeddings(video_embedding, return_tensor=True).float()


                similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())
                reward = similarity_matrix.detach().cpu().numpy()[0][0]

                if self.args.time_reward != 1.0:
                    reward = reward * self.args.time_reward    
                print("sim reward", reward)
                if self.args.threshold_reward:
                    if self.max_sim is not None:
                        if reward < self.max_sim:
                            reward = 0.0
                        else:
                            if self.args.project_reward:
                                total_max = 100
                                # project from max_sim to 100
                                reward = (reward - self.max_sim) / (total_max - self.max_sim) * 100
                    else:
                        raise ValueError("Please provide the max similarity score")
                print("reward", reward)
                if self.args.succ_bonus > 0:
                    if info['success']:
                        reward += self.args.succ_bonus
                reward -= self.args.time_penalty
            return obs, reward, done, info
        
        return obs, -self.args.time_penalty, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0

        if self.args.random_reset:
            self.rank = random.randint(0, 400)
            self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
            self.env = TimeLimit(self.baseEnv, max_episode_steps=128)

        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])


class MetaworldDense(Env):
    def __init__(self, args):
        super(MetaworldDense, self)
        self.door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[args.env_id]
        self.args = args
        self.rank = args.seed
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
                reward += 100

        return obs, reward, done, info
        
    def reset(self):
        self.counter = 0

        if self.args.random_reset:
            self.rank = random.randint(400, 500)
            self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
            # env = door_open_goal_hidden_cls(seed=rank)
            self.env = TimeLimit(self.baseEnv, max_episode_steps=128)
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])




def make_env(args, eval = False):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if not eval:
                env = MetaworldSparse(args)
        else:
            env = MetaworldDense(args)
        env = Monitor(env, os.path.join(log_dir, str(args.seed)))
        return env
    return _init




class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq

    def _on_step(self) -> bool:
        result = super(CustomEvalCallback, self)._on_step()

        if self.video_freq > 0 and self.n_calls % self.video_freq == 0:
            video_buffer, success = self.record_video()
            # wandb.log({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            wandb.log({f"eval/evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, step = self.n_calls)
            wandb.log({f"eval/evaluate_succ": success}, step = self.n_calls)
            print("video logged")

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        success = 0
        for _ in range(128):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode='rgb_array')
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, _, info = self.eval_env.step(action)
            if info['success']:
                success = 1 
                break

        video_buffer = io.BytesIO()

        with imageio.get_writer(video_buffer, format='mp4', fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_buffer.seek(0)
        return video_buffer, success





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
    experiment_name = "s3d_textTrans_TRIPLET" + args.algo + "_" + args.env_id 

    if args.threshold_reward:
        experiment_name = experiment_name + "_Thld"
    if args.project_reward:
        experiment_name = experiment_name + "_ProjReward"
    # if args.norm_input:
    #     experiment_name = experiment_name + "_NormIn"
    # if args.norm_output:
    #     experiment_name = experiment_name + "_NormOut"
    # if args.time_reward != 1.0:
    #     experiment_name = experiment_name + "_XReward" + str(args.time_reward)
    # if args.time:
    #     experiment_name = experiment_name + "_Time"
    # else:
    #     experiment_name = experiment_name + "_NoTime"
    if args.succ_end:
        experiment_name = experiment_name + "_SuccEnd"


    # if args.algo.lower() == 'sac':
    experiment_name = experiment_name + "_" + args.exp_name_end
    run_group = experiment_name 
    experiment_name = experiment_name + "_" + str(args.seed) + "NEW"

    if args.wandb:
        run = wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group=run_group,
            config=args,
            name=experiment_name,
            monitor_gym=True,
            sync_tensorboard=True,
        )


    column1 = ["text_string"]
    table1 = wandb.Table(columns=column1)
    table1.add_data([args.text_string])  

    column2 = ["env_id"]
    table2 = wandb.Table(columns=column2)
    table2.add_data([args.env_id])  
    wandb.log({"text_string": table1, "env_id": table2})


    log_dir = f"/scr/jzhang96/logs/baseline_logs/{experiment_name}"
    # log_dir = f"/home/jzhang96/logs/baseline_logs/{experiment_name}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.n_envs > 1:
        envs = SubprocVecEnv([make_env(args, eval = False) for i in range(args.n_envs)])
    else:
        envs = DummyVecEnv([make_env(args, eval = False)])

    if args.algo.lower() == 'ppo':
        if not args.pretrained:
            model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps,
                        batch_size=args.n_steps * args.n_envs, n_epochs=1, ent_coef=args.entropy_term)
        else:
            model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif args.algo.lower() == 'sac':
        if not args.pretrained:
            model = SAC("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, 
                        # batch_size=args.n_steps * args.n_envs,
                        ent_coef=args.entropy_term , buffer_size=args.total_time_steps, learning_starts=256)
        else:
            model = SAC.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    if args.n_envs > 1:
        eval_env = SubprocVecEnv([make_env(args, eval = True) for i in range(args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    else:
        eval_env = DummyVecEnv([make_env(args, eval = True)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation

    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=log_dir,
                                    log_path=log_dir, eval_freq=args.eval_freq, video_freq=args.video_freq,
                                    deterministic=True, render=False)
     
    wandb_callback = WandbCallback(verbose = 1)
    callback = CallbackList([eval_callback, wandb_callback])
    model.learn(total_timesteps=int(args.total_time_steps), callback=callback)
    model.save(f"{log_dir}/{experiment_name}")


if __name__ == '__main__':
    main()