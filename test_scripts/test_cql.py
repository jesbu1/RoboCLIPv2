import joblib
from gym import Env, spaces
from offline_rl_algorithms.offline_replay_buffers import H5ReplayBuffer
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
import sys

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from metaworld_runs.eval_utils import eval_policys

from offline_rl_algorithms.cql import CQL
from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm


from metaworld_runs.metaworld_envs_xclip_text_transform import SingleLayerMLP, MetaworldSparse, MetaworldDense, parse_entropy_term, make_env, CustomEvalCallback, CustomWandbCallback


class OfflineEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(OfflineEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq

    def _on_step(self) -> bool:
        result = super(OfflineEvalCallback, self)._on_step()

        if self.video_freq > 0 and self.n_calls % self.video_freq == 0:
            video_buffer = self.record_video()
            # wandb.log({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            wandb.log({"eval/evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, step = self.n_calls)
            #wandb.log({f"eval/evaluate_succ": success}, step = self.n_calls)
            print("video logged")

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        #success = 0
        for _ in range(500):  # You can adjust the number of steps for recording
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



class OfflineWandbCallback(WandbCallback):
    # def _on_rollout_end(self):
    #     # Log episode metrics with environment steps as x-axis
    #     wandb.log({
    #         'episode_reward': sum(self.locals['rewards']),  # Cumulative reward for the episode
    #         'episode_length': len(self.locals['rewards'])   # Length of the episode
    #     }, step=self.model.num_timesteps)

    @property
    def wandb_log_step(self) -> int:
        if hasattr(self.model, "offline_num_timesteps"):
            print(self.num_timesteps, self.locals['self'].offline_num_timesteps)
            return self.num_timesteps + self.locals['self'].offline_num_timesteps
            
        return self.num_timesteps
            

    # def _on_rollout_end(self):
    #     # Log episode metrics with environment steps as x-axis
    #     wandb.log({
    #         'episode_reward': sum(self.locals['rewards']),  # Cumulative reward for the episode
    #         'episode_length': len(self.locals['rewards'])   # Length of the episode
    #     }, step=self.model.num_timesteps) 



    def _on_step(self):
        # Log training metrics
        # print done
        #if done and done is True, log the info
        if 'metrics' in self.locals:
            wandb.log(self.locals['metrics'], step = self.wandb_log_step)
        # done_array = self.locals["dones"]
        # infos = self.locals["infos"]
        
    #     for i, done in enumerate(done_array):
    #         if done:

    #             succ = infos[i].get('success', 0)
    #             roboclip_reward = infos[i].get('roboclip_reward', 0)
    #             total_reward = infos[i].get('total_reward', 0)
    #             dense_return = infos[i].get('dense_return', 0)
    #             dense_reward = infos[i].get('dense_reward', 0)
    #             ep_length = infos[i].get('ep_length', 0)
    #             RMS_reward = infos[i].get('RMS_reward', 0)
    #             RMS_total_reward = infos[i].get('RMS_total_reward', 0)
    #             offset = infos[i].get('offset', 0)
    #             print("episode logged", self.wandb_log_step)
    #             wandb.log({"origin_episode_info/episode_success": succ,
    #                         "origin_episode_info/roboclip_reward": roboclip_reward,
    #                         "origin_episode_info/RoboCLIP_bonus_reward": total_reward,
    #                         "origin_episode_info/dense_return": dense_return,
    #                         "origin_episode_info/dense_reward": dense_reward,
    #                         "origin_episode_info/ep_length": ep_length,

    #                         "RMS/RMS_reward": RMS_reward,
    #                         "RMS/RMS_total_reward": RMS_total_reward,
    #                         "RMS/offset": offset
                            
    #                         }, step = self.wandb_log_step)

                
    #     return True



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='cql', choices=['ppo', 'sac', 'cql', 'calibrated_cql'])
    parser.add_argument('--text_string', type=str, default='opening door')
    parser.add_argument('--dir_add', type=str, default='')
    parser.add_argument('--env_id', type=str, default='window-open-v2-goal-hidden')
    parser.add_argument('--offline_training_steps', type=int, default=1000000)
    parser.add_argument('--total_time_steps', type=int, default=1000000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--video_freq', type=int, default=40000)
    parser.add_argument('--succ_end', action="store_true")
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--pca_path', type=str, default=None)
    parser.add_argument('--transform_base_path', type=str, default=None)
    parser.add_argument('--transform_model_path', type=str, default=None)
    parser.add_argument('--random_reset', action="store_true")
    parser.add_argument('--target_gif_path', type=str, default="/scr/jzhang96/metaworld_generate_gifs/")
    #parser.add_argument('--target_gif_path', type=str, default="/home/jzhang96/RoboCLIPv2/metaworld_generate_gifs/")
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
    parser.add_argument('--xclip_model', type=str, default='microsoft/xclip-base-patch16-zero-shot')
    parser.add_argument('--frame_length', type=int, default=32)
    parser.add_argument("--exp_name_end", type=str, default="triplet_hard_neg")
    parser.add_argument("--sparse_only", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--obs_env", action="store_true")


    args = parser.parse_args()
    return args

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
    # if args.pca_path != None:
    #     experiment_name = "ep500_PCA_" + "xclip_textTRANS_" + args.algo + "_" + args.env_id
    # else:
    #     experiment_name = "ep500_NOPCA_" +"xclip_textTRANS_" + args.algo + "_" + args.env_id

    experiment_name = args.algo + "_" + args.env_id
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

    # if args.succ_bonus > 0:
    #     experiment_name = experiment_name + "_SuccBonus" + str(args.succ_bonus)
    # if args.time_penalty > 0:
    #     experiment_name = experiment_name + "_TimePenalty" + str(args.time_penalty)
    # if args.algo.lower() == 'sac':
    # experiment_name = experiment_name + "_Entropy" + str(args.entropy_term)
    experiment_name = experiment_name + args.exp_name_end
    run_group = experiment_name + "NEW"
    # experiment_name = experiment_name + "_" + str(args.seed) + "NEW"
    wandb.disabled = True

    if args.wandb:
        run = wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group=run_group,
            config=args,
            name=experiment_name,
            monitor_gym=True,
            sync_tensorboard=False,
        )


        # column1 = ["text_string"]
        # table1 = wandb.Table(columns=column1)
        # table1.add_data([args.text_string])  

        # column2 = ["env_id"]
        # table2 = wandb.Table(columns=column2)
        # table2.add_data([args.env_id])  
        # wandb.log({"text_string": table1, "env_id": table2})


    # log_dir = f"/scr/jzhang96/logs/baseline_logs/{experiment_name}"
    log_dir = f"logs/baseline_logs/{experiment_name}"
    # log_dir = f"/home/jzhang96/logs/baseline_logs/{experiment_name}"

    args.log_dir = log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.n_envs > 1:
        envs = SubprocVecEnv([make_env(args, eval = False) for i in range(args.n_envs)])
    else:
        envs = DummyVecEnv([make_env(args, eval = False)])

    if args.algo.lower() == 'ppo':
        model_class = PPO
        if not args.pretrained:
            model = model_class("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps,
                        batch_size=args.n_steps * args.n_envs, n_epochs=1, ent_coef=args.entropy_term)
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif args.algo.lower() == 'sac':
        model_class = SAC
        if not args.pretrained:
            model = model_class("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, 
                        # batch_size=args.n_steps * args.n_envs,
                        ent_coef="auto", buffer_size=args.total_time_steps, learning_starts=4000, seed=args.seed)
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif args.algo.lower() in ['cql', 'calibrated_ql']:
        use_calibrated_cql = args.algo.lower() == 'calibrated_ql'
        model_class = CQL
        if not args.pretrained:
            model = model_class("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, 
                        # batch_size=args.n_steps * args.n_envs,
                        ent_coef="auto", buffer_size=args.total_time_steps, learning_starts=4000, seed=args.seed, min_q_weight=5.0, min_q_temp=1.0, use_calibrated_q=use_calibrated_cql, learning_rate=0.0001)
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    if args.n_envs > 1:
        eval_env = SubprocVecEnv([make_env(args, eval = True) for i in range(args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    else:
        eval_env = DummyVecEnv([make_env(args, eval = True)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation

    eval_callback = OfflineEvalCallback(eval_env, best_model_save_path=log_dir,
                                    log_path=log_dir, eval_freq=1, video_freq=1,
                                    deterministic=True, render=False, n_eval_episodes = 25)

    
    
    # wandb_callback = WandbCallback(verbose = 1)
    # callback = CallbackList([eval_callback, wandb_callback])
    if args.wandb:
        # customwandbcallback = CustomWandbCallback()
        customwandbcallback = OfflineWandbCallback()
        callback = CallbackList([eval_callback, customwandbcallback])
    else:
        callback = eval_callback

    # load the offline replay buffer
    if isinstance(model, OfflineRLAlgorithm):
        h5_path = "updated_trajs.h5"
        buffer = H5ReplayBuffer(h5_path)
        model.learn_offline(offline_replay_buffer=buffer, train_steps=args.offline_training_steps, callback=callback)
    # once learn offline is done, fix the eval callback
    eval_callback.eval_freq = args.eval_freq
    eval_callback.video_freq = args.video_freq


    model.learn(total_timesteps=int(args.total_time_steps), callback=callback)
    model.save(f"{log_dir}/{experiment_name}")

    # Evaluate the agent
    # load the best model
    model = model_class.load(f"{log_dir}/best_model")
    success_rate = eval_policys(args, MetaworldDense, model)

    if args.wandb:
        wandb.log({"eval_SR/evaluate_succ": success_rate}, step = 0)


if __name__ == '__main__':
    main()