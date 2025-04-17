import random

import numpy as np
import torch as th
from gym import Env
import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
)

from envs.wrappers import *
from reward_model.env_reward_model import EnvRewardModel

environment_to_instruction = {
    "assembly-v2": "assembling",
    "basketball-v2": "playing basketball",
    "bin-picking-v2": "picking bin",
    "box-close-v2": "closing box",
    "button-press-topdown-v2": "pressing button from top",
    "button-press-topdown-wall-v2": "pressing button",
    "button-press-v2": "pressing button from side",
    "button-press-wall-v2": "pressing button from side",
    "coffee-button-v2": "pressing coffee button",
    "coffee-pull-v2": "pulling cup",
    "coffee-push-v2": "pushing coffee cup",
    "dial-turn-v2": "turning dial",
    "disassemble-v2": "disassembling",
    "door-close-v2": "closing door",
    "door-lock-v2": "locking door",
    "door-open-v2": "opening door",
    "door-unlock-v2": "unlocking door",
    "hand-insert-v2": "inserting bin",
    "drawer-close-v2": "closing drawer",
    "drawer-open-v2": "opening drawer",
    "faucet-open-v2": "opening faucet",
    "faucet-close-v2": "closing faucet",
    "hammer-v2": "hammering nail",
    "handle-press-side-v2": "pressing handle from side",
    "handle-press-v2": "pressing handle",
    "handle-pull-side-v2": "pulling handle",
    "handle-pull-v2": "pulling handle",
    "lever-pull-v2": "pulling lever",
    "peg-insert-side-v2": "inserting peg",
    "pick-place-wall-v2": "placing bin to shelf",
    "pick-out-of-hole-v2": "picking bin",
    "reach-v2": "reaching red",
    "push-back-v2": "pulling bin back",
    "push-v2": "pushing block",
    "pick-place-v2": "placing bin to shelf",
    "plate-slide-v2": "sliding plate",
    "plate-slide-side-v2": "sliding plate",
    "plate-slide-back-v2": "sliding plate",
    "plate-slide-back-side-v2": "sliding plate",
    "peg-unplug-side-v2": "unpluging peg",
    "soccer-v2": "kicking soccer ball",
    "stick-push-v2": "pushing stick",
    "stick-pull-v2": "pulling stick",
    "push-wall-v2": "pushing bin",
    "reach-wall-v2": "reaching red",
    "shelf-place-v2": "placing bin to shelf",
    "sweep-into-v2": "sweep blocks into hole",
    "sweep-v2": "sweeping block",
    "window-open-v2": "opening window",
    "window-close-v2": "closing window",
}

instruction_to_environment = {v: k for k, v in environment_to_instruction.items()}


# Define a base environment for MetaWorld
class MetaworldBase(Env):
    def __init__(
        self,
        env_id,
        seed=0,
        goal_observable=False,
        random_reset="train",
        max_episode_steps=128,
        use_proprio=False,
    ):
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
        # print(self.all_env_types, env_id)
        if goal_observable:
            env_id = env_id + "-goal-observable"
            self.base_env = self.all_env_types[env_id](seed=seed)
        else:
            env_id = env_id + "-goal-hidden"
            self.base_env = self.all_env_types[env_id](seed=seed)

        self.max_episode_steps = max_episode_steps

        self.base_env = TimeLimit(
            self.base_env, max_episode_steps=self.max_episode_steps
        )

        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.image_keys = ["image"]
        self.image_reward_idx = 0

        self.observation_space = gym.spaces.Dict(
            {
                "proprio": gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
            }
        )

        self.rank = seed
        self.env_id = env_id
        self.random_reset = random_reset

        self.use_proprio = use_proprio

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
        state, reward, done, info = self.base_env.step(action)

        obs = self.get_obs(state)

        # if success, we add "is_success" to the info
        if "success" in info and info["success"]:
            info["is_success"] = True
        else:
            info["is_success"] = False

        return obs, reward, done, info

    def get_obs(self, state):
        """
        Get the current observation of the environment.

        Returns:
            observation (object): agent's observation of the current environment
        """
        # state = self.base_env._get_obs(self.base_env.prev_time_step)
        obs = {}
        if self.use_proprio:
            obs["proprio"] = state[:4]

        if self.image_keys:
            image = self.render(mode="rgb_array")
            obs["image"] = image

        return obs

    def reset(self):
        """
        Resets the environment and optionally resets the underlying environment with a random seed.

        Returns:
            observation (object): the initial observation
        """
        if self.random_reset == "train":
            self.rank = random.randint(100, 400)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(
                self.base_env, max_episode_steps=self.max_episode_steps
            )
        elif self.random_reset == "eval":
            self.rank = random.randint(400, 500)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(
                self.base_env, max_episode_steps=self.max_episode_steps
            )
        elif self.random_reset == "demo":
            self.rank = random.randint(0, 100)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(
                self.base_env, max_episode_steps=self.max_episode_steps
            )

        state = self.base_env.reset()

        obs = self.get_obs(state)

        return obs

    def render(self, mode="rgb_array"):
        """
        Render the environment.

        Returns:
            observation (object): the current observation
        """
        return self.base_env.render(mode)

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


class MetaworldImageEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super(MetaworldImageEmbeddingWrapper, self).__init__(env)
        self.reward_model = reward_model

        # The observation space is a dict
        # Let us add image_feature to the observation space

        current_obs_space = self.env.observation_space
        assert isinstance(current_obs_space, spaces.Dict), (
            "Observation space must be a Dict."
        )

        image_keys = self.env.image_keys

        # Define the new observation space
        new_spaces = current_obs_space.spaces.copy()
        # Add a new key for the image feature corresponding to each image key
        new_spaces["image_feature_0"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(reward_model.img_output_dim,),
            dtype=np.float32,
        )

        # Set the updated observation space
        self.observation_space = spaces.Dict(new_spaces)

    def __getstate__(self):
        """Custom method for pickling - exclude reward_model which might contain unpicklable objects"""
        state = self.__dict__.copy()
        # Remove the reward_model which might not be picklable
        if "reward_model" in state:
            del state["reward_model"]
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)
        # Set reward_model to None - it will need to be set again after unpickling
        self.reward_model = None

    def _observation(self, observation):
        # image = observation["image"]
        # observation["image"] = image
        image = observation["image"]
        image = image[None, None, :, :, :]
        image_feature = self.reward_model.encode_images(image).squeeze()
        observation["image_feature_0"] = image_feature

        return observation

    def reset(self):
        obs = self.env.reset()
        obs = self._observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info

    def seed(self, seed=None):
        pass


# Example usage of the base environment and wrappers
def create_wrapped_env(
    env_id,
    reward_model,
    pca_model=None,
    language_features=None,
    use_time=False,
    monitor=False,
    goal_observable=False,
    success_bonus=0.0,
    is_state_based=False,
    mode="train",
    use_proprio=False,
    dense_rewards_at_end=False,
    action_chunk_size=1,
    logger=None,
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
        monitor: Whether to monitor the environment returns, rewards, etc. (default=False).

    Returns:
        A function that returns the wrapped environment when called.
    """

    def _init():
        if mode == "eval":
            base_env = MetaworldBase(
                env_id,
                goal_observable=goal_observable,
                random_reset="eval",
                use_proprio=use_proprio,
            )
        elif mode == "train":
            base_env = MetaworldBase(
                env_id,
                goal_observable=goal_observable,
                random_reset="train",
                use_proprio=use_proprio,
            )
        elif mode == "demo":
            base_env = MetaworldBase(
                env_id,
                goal_observable=goal_observable,
                random_reset="demo",
                use_proprio=use_proprio,
            )
        else:
            raise ValueError("Invalid mode")

        if pca_model is not None:
            base_env = PCAReducerWrapper(base_env, pca_model)

        if use_time:
            base_env = TimeWrapper(base_env)

        # breakpoint()
        # This replaces the metaworld state-based input with an image embedding too

        dense_eval = True if (mode == "eval" or mode == "demo") else False

        base_env = MetaworldImageEmbeddingWrapper(base_env, reward_model)

        base_env = LearnedRewardWrapper(
            base_env,
            reward_model,
            is_state_based=is_state_based,
            language_features=language_features,
            dense_eval=dense_eval,
        )

        # This adds the language features to the observation
        if language_features is not None:
            base_env = LanguageWrapper(base_env, language_features)

        # Environment keeps an aggregate reward at each step and outputs it only when the episode ends
        if dense_rewards_at_end:
            base_env = RewardAtEndWrapper(base_env)

        base_env = FlattenDictObservationWrapper(base_env, use_proprio=use_proprio)

        if action_chunk_size > 1:
            base_env = ActionChunkingWrapper(
                base_env, chunk_size=action_chunk_size, n_action_steps=action_chunk_size
            )

        # else:
        #     # Then we are an EnvRewardModel
        #     if reward_model.name == 'sparse':
        #         use_sparse = True
        #     elif reward_model.name == 'dense':
        #         use_sparse = False
        #     base_env = RewardWrapper(base_env, sparse=use_sparse, success_bonus=reward_model.success_bonus)

        if monitor:
            base_env = Monitor(base_env)

        if logger is not None:
            base_env = LoggingWrapper(base_env, logger, prefix=mode)

        return base_env

    return _init


if __name__ == "__main__":
    env = MetaworldBase("door-open-v2", goal_observable=True)
    env.reset()
    env.render()
