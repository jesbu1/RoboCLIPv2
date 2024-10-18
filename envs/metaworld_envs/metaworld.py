import random

import numpy as np
import torch as th
from gym import Env
from gym.wrappers.time_limit import TimeLimit
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
)

from envs.metaworld_envs.wrappers import *


# Define a base environment for MetaWorld
class MetaworldBase(Env):
    def __init__(self, env_id, seed=0, goal_observable=False, random_reset=False):
        """
        Parameters
        ----------
        env_id : int
            index of the environment
        seed : int
            random seed
        goal_observable : bool
            whether the goal is observable
        random_reset : bool
            whether to randomly reset the environment
        """
        super(MetaworldBase, self).__init__()

        self.all_env_types = (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
            if goal_observable
            else ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
        )
        if goal_observable:
            self.base_env = self.all_env_types[env_id](seed=seed)
        else:
            self.base_env = self.all_env_types[env_id](seed=seed)

        self.base_env = TimeLimit(self.base_env, max_episode_steps=500)

        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.rank = seed
        self.env_id = env_id
        self.random_reset = random_reset

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes for learning)
        """
        obs, reward, done, info = self.base_env.step(action)
        return obs, reward, done, info

    def get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            observation (object): agent's observation of the current environment
        """
        return self.base_env._get_obs(self.base_env.prev_time_step)

    def reset(self):
        """
        Resets the environment and optionally resets the underlying environment with a random seed.

        Returns:
            observation (object): the initial observation
        """
        if self.random_reset:
            self.rank = random.randint(0, 400)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(self.base_env, max_episode_steps=500)

        return self.base_env.reset()

    def render(self):
        """
        Render the environment.

        Returns:
            observation (object): the current observation
        """
        return self.base_env.render()

    # def warm_up_run(self):
    #     self.env.reset()
    #     images = []
    #     frame_num = random.randint(32, 128)

    #     for _ in range(frame_num):
    #         action = self.env.action_space.sample()
    #         _, _, _, _ = self.env.step(action)
    #         images.append(self.env.render()[:, :, :3])
    #     images = np.array(images)

    #     with th.no_grad():
    #         frames = adjust_frames_xclip(
    #             images,
    #             target_frame_count=self.args.frame_length,
    #             processor=self.processor,
    #         ).cuda()
    #         frames = self.net.get_video_features(frames)

    #     return frames

    def close(self):
        """
        Closes the environment. This is used to clean up resources and shutdown any child processes.

        Returns:
            None
        """
        return self.base_env.close()


# Example usage of the base environment and wrappers
def create_wrapped_env(
    env_id,
    pca_model=None,
    language_features=None,
    success_bonus=0.0,
    use_simulator_reward=False,
    use_time=True,
):
    """
    Creates a wrapped MetaWorld environment with the given options.

    Args:
        env_id: The MetaWorld environment ID.
        pca_model: The PCA model to use for dimensionality reduction (optional).
        language_features: The language features to use for the environment (optional).
        sparse_reward: Whether to use sparse rewards (default=True).
        use_simulator_reward: Whether to use the simulator reward (default=False).
        use_time: Whether to add time to the observation (default=True).

    Returns:
        A function that returns the wrapped environment when called.
    """
    def _init():
        base_env = MetaworldBase(env_id)

        if pca_model is not None:
            base_env = PCAReducerWrapper(base_env, pca_model)

        if language_features is not None:
            base_env = LanguageWrapper(base_env, language_features)

        if use_time:
            base_env = TimeWrapper(base_env)

        use_sparse_only = not use_simulator_reward
        base_env = RewardWrapper(base_env, sparse=use_sparse_only, success_bonus=success_bonus)

        return base_env

    return _init
