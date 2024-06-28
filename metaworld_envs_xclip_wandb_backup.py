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
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoProcessor 
from gym.wrappers import RecordVideo
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import io


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text_string', type=str, default='Closing door')
    parser.add_argument('--dir_add', type=str, default='')
    parser.add_argument('--env_id', type=str, default='door-close-v2-goal-hidden')
    parser.add_argument('--env_type', type=str, default='sparse_learnt')
    parser.add_argument('--total_time-steps', type=int, default=200000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=512)
    parser.add_argument('--video_freq', type=int, default=2048)

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
            for _ in range (3):
                print("text_string", text_string)
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
            env = MetaworldSparse(env_id=env_id, text_string=args.text_string, time=True, rank=rank)
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




class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq

    def _on_step(self) -> bool:
        result = super(CustomEvalCallback, self)._on_step()

        if self.video_freq > 0 and self.n_calls % self.video_freq == 0:
            video_buffer = self.record_video()

            wandb.log({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")})
        #     video_path = os.path.join(self.best_model_save_path, f"evaluation_video_{self.n_calls}.mp4")
        #     self.record_video(video_path)
        #     wandb.log({f"evaluation_video_{self.n_calls}": wandb.Video(video_path, fps=4, format="mp4")})

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        for _ in range(128):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode='rgb_array')
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, _, _ = self.eval_env.step(action)
            # if done:
            #     break

        # Save the video
        # height, width, _ = frames[0].shape
        video_buffer = io.BytesIO()
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(video_buffer, fourcc, 20, (width, height))

        # for frame in frames:
        #     out.write(frame)
        # out.release()

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
    


    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    # wandb_eval_task_name = "_".join([str(i) for i in eval_tasks])
    # experiment_name = args.experiment_name + "_" + wandb_eval_task_name + "_" + str(args.seed)
    # if args.mse:
    #     experiment_name = experiment_name + "_mse_" + str(args.mse_weight)
    experiment_name = "xclip-wandb_" + args.env_id + "_" + str(args.seed)

    if args.wandb:
        run = wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group="x-clip-roboclipv1-train" + args.env_id,
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




    log_dir = f"/scr/jzhang96/logs/{experiment_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, i) for i in range(args.n_envs)])

    if not args.pretrained:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, i) for i in range(10, 10+args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation
    # eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                              log_path=log_dir, eval_freq=200,
    #                              deterministic=True, render=False)



    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=log_dir,
                                    log_path=log_dir, eval_freq=args.eval_freq, video_freq=args.video_freq,
                                    deterministic=True, render=False)



                                 
    wandb_callback = WandbCallback(verbose = 1)
    callback = CallbackList([eval_callback, wandb_callback])



    model.learn(total_timesteps=int(args.total_time_steps), callback=callback)
    model.save(f"{log_dir}/trained")







if __name__ == '__main__':
    main()
