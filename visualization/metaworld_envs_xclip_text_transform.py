import joblib
from gym import Env, spaces
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
import torch as th
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
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from eval_utils import eval_policys

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

id_task = json.load(open("id_task.json", "r"))


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


def adjust_frames_xclip(frames, target_frame_count=32, processor=None):
    """
    Ensures same numbers of frames(32). returns a numpy array of shape (target_frame_count, 224, 224, 3)
    """
    frames = np.array(frames)
    frame_count = frames.shape[0]
    # print(f"frames number{frame_count}")
    # frames = th.from_numpy(frames)

    if len(frames) > target_frame_count:
        index = np.linspace(0, len(frames) - 1, target_frame_count, dtype=int)
        frames = frames[index]
    elif len(frames) < target_frame_count:
        last_frame = frames[-1]
        last_frame = np.expand_dims(last_frame, axis=0)
        for _ in range(target_frame_count - len(frames)):
            frames = np.concatenate([frames, last_frame])
    frames = frames[:, 240 - 112:240 + 112, 320 - 112:320 + 112, :]
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
    parser.add_argument('--pca_path', type=str, default=None)
    parser.add_argument('--transform_base_path', type=str, default=None)
    parser.add_argument('--transform_model_path', type=str, default=None)
    parser.add_argument('--random_reset', action="store_true")
    parser.add_argument('--target_gif_path', type=str, default="/scr/jzhang96/metaworld_generate_gifs/")
    # parser.add_argument('--target_gif_path', type=str, default="/home/jzhang96/RoboCLIPv2/metaworld_generate_gifs/")
    parser.add_argument('--time', action="store_false")
    parser.add_argument('--frame_num', type=int, default=32)
    parser.add_argument('--train_orcale', action="store_true")  # load latent from h5 file
    parser.add_argument('--warm_up_runs', type=int, default=0)
    parser.add_argument('--project_reward', action="store_true")
    parser.add_argument('--norm_input', action="store_true")
    parser.add_argument('--norm_output', action="store_true")
    parser.add_argument('--time_reward', type=float, default=1.0)
    parser.add_argument('--threshold_reward', action="store_true")
    parser.add_argument('--entropy_term', type=parse_entropy_term, default="auto")
    parser.add_argument('--time_penalty', type=float, default=0.0)
    parser.add_argument('--succ_bonus', type=float, default=0.0)
    parser.add_argument('--xclip_model', type=str, default='microsoft/xclip-base-patch16-zero-shot')
    parser.add_argument('--frame_length', type=int, default=32)
    parser.add_argument("--exp_name_end", type=str, default="triplet_hard_neg")
    parser.add_argument("--sparse_only", action="store_true")
    parser.add_argument("--baseline", action="store_true")

    args = parser.parse_args()
    return args


class MetaworldSparse(Env):
    # def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
    def __init__(self, args):
        super(MetaworldSparse, self)
        self.args = args
        self.door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[args.env_id]
        self.rank = args.seed
        self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
        self.env = TimeLimit(self.baseEnv, max_episode_steps=128)
        self.time = args.time
        if args.pca_path != None:
            pca_text_path = os.path.join(args.pca_path,
                                         'pca_model_text.pkl')  # '/scr/yusenluo/RoboCLIP/visualization/saved_model/pca_loss_models/[4, 13, 19, 36, 48]_Seed_42/512_linear/pca_model_text.pkl'
            pca_video_path = os.path.join(args.pca_path,
                                          'pca_model_video.pkl')  # '/scr/yusenluo/RoboCLIP/visualization/saved_model/pca_loss_models/[4, 13, 19, 36, 48]_Seed_42/512_linear/pca_model_video.pkl'
            # print(pca_video_path)
            pca_text_model = joblib.load(pca_text_path)
            pca_video_model = joblib.load(pca_video_path)
            self.pca_text_model = pca_text_model
            self.pca_video_model = pca_video_model
            # self.computed_matrix = th.from_numpy(np.dot(self.pca_video_model.components_, self.pca_text_model.components_.T)).float()

        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0] + 1,),
                                         dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []

        self.frame_length = args.frame_length

        if not args.sparse_only:

            with th.no_grad():
                model_name = args.xclip_model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.net = AutoModel.from_pretrained(model_name).cuda()
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.net = self.net.eval()

                self.transform_model = SingleLayerMLP(512, 512, normalize=True)

                transform_model_path = os.path.join(args.transform_base_path, args.transform_model_path)
                dict = th.load(transform_model_path)
                if 'model_state_dict' in dict.keys():
                    self.transform_model.load_state_dict(dict["model_state_dict"])
                else:
                    self.transform_model.load_state_dict(dict)

                self.transform_model = self.transform_model.eval().cuda()
                # self.transform_model.load_state_dict(th.load("/scr/jzhang96/triplet_text_loss_models/triplet_loss_50_42_xclip_TimeShort_Normtriplet/55.pth"))

                self.target_embedding = None

                if args.text_string:
                    for _ in range(3):
                        print("text_string", args.text_string)
                    text_tokens = self.tokenizer([args.text_string], return_tensors="pt")
                    for key in text_tokens:
                        text_tokens[key] = text_tokens[key].cuda()
                    self.target_embedding = self.net.get_text_features(**text_tokens)
                    self.target_embedding = normalize_embeddings(self.target_embedding)
                    if args.pca_path != None:
                        self.target_embedding = th.from_numpy(
                            self.pca_text_model.transform(self.target_embedding.cpu())).cuda()
                    # with th.no_grad():
                    #     if args.pca_path != None:
                    #         self.transform_model.linear.weight = nn.Parameter(
                    #             self.computed_matrix.T.to(dtype=th.float32).cuda())
                    #         self.transform_model.linear.bias = nn.Parameter(
                    #             th.zeros(self.target_embedding.shape[1], dtype=th.float32).cuda())
                    #     self.transform_model = self.transform_model.eval().cuda()

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

    def render(self):
        frame = self.env.render()
        return frame

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter / 128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])

        if self.args.succ_end:
            if info['success']:
                done = True

        if done:
            if self.args.sparse_only:
                if info['success']:
                    reward = 1.0 * self.args.time_reward
                else:
                    reward = 0.0

            else:
                with th.no_grad():
                    frames = adjust_frames_xclip(self.past_observations, target_frame_count=self.args.frame_length,
                                                 processor=self.processor).cuda()
                    video_embedding = self.net.get_video_features(frames)
                    # used to test the model with norm embeddings
                    if self.args.norm_output:
                        video_embedding = normalize_embeddings(video_embedding, return_tensor=True).float()
                        self.target_embedding = normalize_embeddings(self.target_embedding, return_tensor=True).float()

                    if args.pca_path != None:
                        video_embedding = th.from_numpy(
                            self.pca_video_model.transform(video_embedding.cpu())).float().cuda()

                    # print(f"video_embedding dtype: {video_embedding.dtype}")
                    # print(f"transform_model.linear.weight dtype: {self.transform_model.linear.weight.dtype}")
                    if args.baseline:
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
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0] + 1,),
                                         dtype=np.float32)
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
        obs, _, done, info = self.env.step(action)
        self.counter += 1
        self.counter_total += 1
        t = self.counter / 128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])

        reward = 0.0

        if self.args.succ_end:
            if info['success']:
                done = True
        if info["success"]:
            reward += 1

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

    def reset_seed(self, seed):
        self.counter = 0

        self.rank = seed
        self.baseEnv = self.door_open_goal_hidden_cls(seed=self.rank)
        self.env = TimeLimit(self.baseEnv, max_episode_steps=128)
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])


def make_env(args, eval=False):
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


class CustomWandbCallback(WandbCallback):
    def _on_rollout_end(self):
        # Log episode metrics with environment steps as x-axis
        wandb.log({
            'episode_reward': sum(self.locals['rewards']),  # Cumulative reward for the episode
            'episode_length': len(self.locals['rewards'])  # Length of the episode
        }, step=self.model.num_timesteps)


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq

    def _on_step(self) -> bool:
        result = super(CustomEvalCallback, self)._on_step()

        if self.video_freq > 0 and self.n_calls % self.video_freq == 0:
            video_buffer = self.record_video()
            # wandb.log({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            wandb.log({f"eval/evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, step=self.n_calls)
            # wandb.log({f"eval/evaluate_succ": success}, step = self.n_calls)
            print("video logged")

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        # success = 0
        for _ in range(128):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode='rgb_array')
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, _, info = self.eval_env.step(action)
            # print(type(info))
            # print(info)
            # if info['success']:
            #     success = 1
            #     break

        video_buffer = io.BytesIO()

        with imageio.get_writer(video_buffer, format='mp4', fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_buffer.seek(0)
        return video_buffer


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
    if args.pca_path != None:
        experiment_name = "PCA_" + "xclip_textTRANS_" + args.algo + "_" + args.env_id
    else:
        experiment_name = "NOPCA_" + "xclip_textTRANS_" + args.algo + "_" + args.env_id
    if args.train_orcale:
        experiment_name = experiment_name + "_Oracle"
    if args.threshold_reward:
        experiment_name = experiment_name + "_Thld"
    if args.project_reward:
        experiment_name = experiment_name + "_ProjReward"
    # if args.norm_input:
    #     experiment_name = experiment_name + "_NormIn"
    # if args.norm_output:
    #     experiment_name = experiment_name + "_NormOut"
    experiment_name = experiment_name + '_PCA_512'
    # if args.time_reward != 1.0:
    #     experiment_name = experiment_name + "_XReward" + str(args.time_reward)
    # if args.time:
    #     experiment_name = experiment_name + "_Time"
    # else:
    #     experiment_name = experiment_name + "_NoTime"
    if args.succ_end:
        experiment_name = experiment_name + "_SuccEnd"
    # if args.random_reset:
    #     experiment_name = experiment_name + "_RandReset"

    if args.succ_bonus > 0:
        experiment_name = experiment_name + "_SuccBonus" + str(args.succ_bonus)
    # if args.time_penalty > 0:
    #     experiment_name = experiment_name + "_TimePenalty" + str(args.time_penalty)
    # if args.algo.lower() == 'sac':
    # experiment_name = experiment_name + "_Entropy" + str(args.entropy_term)
    experiment_name = experiment_name + args.exp_name_end
    run_group = "PCA_Ini" + experiment_name + "NEW"
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

    log_dir = f"/scr/yusenluo/RoboCLIP/visualization/xclip_text_transform_logs/{experiment_name}"
    # log_dir = f"/home/jzhang96/logs/baseline_logs/{experiment_name}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.n_envs > 1:
        envs = SubprocVecEnv([make_env(args, eval=False) for i in range(args.n_envs)])
    else:
        envs = DummyVecEnv([make_env(args, eval=False)])

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
                        ent_coef=args.entropy_term, buffer_size=args.total_time_steps, learning_starts=256, seed=args.seed)
        else:
            model = SAC.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    if args.n_envs > 1:
        eval_env = SubprocVecEnv(
            [make_env(args, eval=True) for i in range(args.n_envs)])  # KitchenEnvDenseOriginalReward(time=True)
    else:
        eval_env = DummyVecEnv([make_env(args, eval=True)])  # KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation

    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=log_dir,
                                       log_path=log_dir, eval_freq=args.eval_freq, video_freq=args.video_freq,
                                       deterministic=True, render=False, n_eval_episodes=25)

    # wandb_callback = WandbCallback(verbose = 1)
    # callback = CallbackList([eval_callback, wandb_callback])
    customwandbcallback = CustomWandbCallback()
    callback = CallbackList([eval_callback, customwandbcallback])
    model.learn(total_timesteps=int(args.total_time_steps), callback=callback)
    model.save(f"{log_dir}/{experiment_name}")

    # Evaluate the agent
    # load the best model
    model = SAC.load(f"{log_dir}/best_model")
    success_rate = eval_policys(args, MetaworldDense, model)
    wandb.log({"eval/evaluate_succ": success_rate})


if __name__ == '__main__':
    main()