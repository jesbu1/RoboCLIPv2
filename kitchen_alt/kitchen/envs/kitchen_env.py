import numpy as np
from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveHingeSlideV0, KitchenKettleV0, KitchenMicrowaveV0, KitchenLightV0
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
from gym import Env, spaces
from gym.spaces import Box
import torch as th



class KitchenEnvSparseOriginalRewardImage(Env):
    def __init__(self, time=True):
        super(KitchenEnvSparseOriginalRewardImage, self)
        env = KitchenMicrowaveHingeSlideV0()
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        self.observation_space = Box(low=0, high=255, shape=(250, 250, 3), dtype=np.uint8)
        self.action_space = self.env.action_space
        self.episode_reward = 0.0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = self.render()
        self.episode_reward += r
        # self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        if done:
            return obs, self.episode_reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.episode_reward = 0.0
        # self.past_observations = []
        self.env.reset()
        return self.render()
class KitchenEnvSparseOriginalReward(Env):
    def __init__(self, time=True):
        super(KitchenEnvSparseOriginalReward, self)
        env = KitchenMicrowaveHingeSlideV0()
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.episode_reward = 0.0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        t = info["time"]
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        self.episode_reward += r
        # self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        if done:
            return obs, self.episode_reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.episode_reward = 0.0
        # self.past_observations = []
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])

class KitchenEnvDenseOriginalReward(Env):
    def __init__(self, time=True):
        super(KitchenEnvDenseOriginalReward, self)
        env = KitchenKettleV0()
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.episode_reward = 0.0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        t = info["time"]
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        self.episode_reward += r
        # self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        return obs, r, done, info

    def reset(self):
        self.episode_reward = 0.0
        # self.past_observations = []
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])
        
class KitchenEnvSparseReward(Env):
    def __init__(self, text_string="robot opening door", time=True):
        super(KitchenEnvSparseReward,self)
        env = KitchenMicrowaveHingeSlideV0()
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('/lab/ssontakk/S3D_HowTo100M/cem_planning/s3d_dict.npy', 512)

        # Load the model weights
        self.net.load_state_dict(th.load('/lab/ssontakk/S3D_HowTo100M/cem_planning/s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        text_output = self.net.text_module([text_string])
        self.target_embedding = text_output['text_embedding']
        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)


    def preprocess_kitchen(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        frames = frames[:, :,::4,:,:]
        if np.equal(np.mod(frames, 1).all(), 0):
            frames = frames/255
        return frames
        
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        t = info["time"]
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_kitchen(self.past_observations)
            
        
        
            video = th.from_numpy(frames)
            # print("video.shape", video.shape)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']
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
