from gym import Env, spaces
from gym.spaces import Box

import numpy as np
from stable_baselines3 import SAC
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

import os
from stable_baselines3.common.monitor import Monitor
from memory_profiler import profile
import argparse
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from matplotlib import animation
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo
import imageio


class MetaworldDemo(Env):
    def __init__(self, env_id, rank=0):
        super(MetaworldDemo,self)
        self.task = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id]
        self.rank = rank
        env = self.task(seed=self.rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.action_space = self.env.action_space
        # self.succ = False
        self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0],), dtype=np.float32)
        self.counter = 0
        # import ipdb ; ipdb.set_trace()
    
    def render(self):
        frame = self.env.render()
        center = 240, 320
        # h, w = (250, 250)
        h, w = (224, 224)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # self.succ = info['success']
        # self.counter += 1
        # if self.counter >= 128:
        #     done = True
        # elif info['success'] == 1:
        #     done = True
        # else:
        #     done = False

        return obs, reward, done, info

    def reset(self):
        self.rank += 1
        new_env = self.task(seed = self.rank)
        self.env = TimeLimit(new_env, max_episode_steps=128)
        # self.counter = 0

        return self.env.reset()

    def render(self, mode=True):
        return self.env.render()





class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, video_folder: str, video_length: int = 128, deterministic: bool = True, render: bool = True):
        super(VideoRecorderCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.video_folder = video_folder
        self.video_length = video_length
        self.deterministic = deterministic
        self.render = render
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.video_folder is not None:
            os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.record_video()

        return True

    def record_video(self):
        env = self.eval_env.envs[0] 
        # succ = getattr(env, 'succ', 'unknown')
        rank = getattr(env, 'rank', 'unknown')
        env.render(mode=True)
        
        # self.eval_env = VecVideoRecorder(self.eval_env, video_folder=self.video_folder, record_video_trigger=lambda x: x == 0, video_length=self.video_length, name_prefix=video_file)
        # mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=1, deterministic=self.deterministic, render=self.render)
        
        obs = self.eval_env.reset()
        img_buffer = [self.eval_env.render(mode=True)]
        sum_reward = 0
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, info = self.eval_env.step(action)
            # import ipdb ; ipdb.set_trace()
            img_buffer.append(self.eval_env.render(mode=True))
            finish = (info[0]["success"] == 1)
            sum_reward += reward
            if finish:
                obs = self.eval_env.reset()
                break
        succ = "True" if info[0]["success"] == 1 else "False"
        # save 2 digits
        sum_reward = np.round(sum_reward, 2)
        video_file = f"step_{self.num_timesteps}_seed_{rank}_sumreward{sum_reward}_succ_{succ}.mp4"
        video_path = os.path.join(self.video_folder, video_file)
        imageio.mimsave(video_path, img_buffer, fps=15)

        self.eval_env.close()








def main(args):
    # set seed
    np.random.seed(args.seed)

#     # generate 50 random index
#     random_index = np.random.randint(0, 50, 50)
    print("task_id: ", args.task_id)
    tasks = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
    select_task = tasks[args.task_id]
    env = MetaworldDemo(select_task)

    video_save_path = os.path.join(args.output_dir, "videos_"+str(select_task))
    policy_save_path = os.path.join(args.output_dir, "policies_"+str(select_task))
    log_save_path = os.path.join(args.output_dir, "logs_"+str(select_task))

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    if not os.path.exists(policy_save_path):
        os.makedirs(policy_save_path)

    model = SAC('MlpPolicy', env, verbose=1)
    env = DummyVecEnv([lambda: env])

    video_callback = VideoRecorderCallback(eval_env=env, eval_freq=args.eval_freq, video_folder=video_save_path, video_length=128)
    model.learn(total_timesteps=args.train_steps, callback=video_callback)




if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/scr/jzhang96/metaworld_generation_wo_done")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=1024000)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=2560)
    args = parser.parse_args()

    main(args)
