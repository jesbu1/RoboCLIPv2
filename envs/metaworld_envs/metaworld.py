import random

import numpy as np
import torch as th
from gym import Env
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from memory_profiler import profile
from envs.metaworld_envs.wrappers import *
from models.reward_model.env_reward_model import EnvRewardModel
from gym.wrappers.normalize import NormalizeReward

environment_to_instruction = {
    "assembly-v2": "assembling",
    "basketball-v2": "playing basketball",
    "bin-picking-v2": "picking bin",
    "box-close-v2": "closing box",
    "button-press-topdown-v2": "Press the button from top",
    "button-press-topdown-wall-v2": "Press the button from top",
    "button-press-v2": "Press the button from side",
    "button-press-wall-v2": "Press the button from side",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "coffee-push-v2": "Push the coffee cup",
    "dial-turn-v2": "Turn the dial",
    "disassemble-v2": "disassembling",
    "door-close-v2": "Close the door",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "door-open-v2": "Open the door",
    "door-unlock-v2": "Turn door lock clockwise",
    "hand-insert-v2": "Pick up the block and insert it into the hole",
    "drawer-close-v2": "Close the drawer",
    "drawer-open-v2": "opening drawer",
    "faucet-open-v2": "Open the faucet",
    "faucet-close-v2": "Close the faucet",
    "hammer-v2": "hammering nail",
    "handle-press-side-v2": "Press the handle from side",
    "handle-press-v2": "Press the handle",
    "handle-pull-side-v2": "Pull the handle up from the side",
    "handle-pull-v2": "Pull the handle",
    "lever-pull-v2": "pulling lever",
    "peg-insert-side-v2": "Insert the peg",
    "pick-place-wall-v2": "Pick up the block and placing it to the goal position",
    "pick-out-of-hole-v2": "picking bin",
    "reach-v2": "Reach the goal",
    "push-back-v2": "Push the block back to the goal",
    "push-v2": "Push the block to the goal",
    "pick-place-v2": "Pick up the block and placing it to the goal position",
    "plate-slide-v2": "Slide the plate into the gate",
    "plate-slide-side-v2": "Slide the plate into the gate from the side",
    "plate-slide-back-v2": "Slide the plate out of the gate",
    "plate-slide-back-side-v2": "Slide the plate out of the gate from the side",
    "peg-unplug-side-v2": "unpluging peg",
    "soccer-v2": "Slide the ball into the gate",
    "stick-push-v2": "Push the stick",
    "stick-pull-v2": "Pull the stick",
    "push-wall-v2": "pushing bin",
    "reach-wall-v2": "Reach the goal",
    "shelf-place-v2": "placing bin to shelf",
    "sweep-into-v2": "Sweep the block into the hole",
    "sweep-v2": "sweeping block",
    "window-open-v2": "Open the window",
    "window-close-v2": "Close the window",
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
        terminate_on_success=False,
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
        self.rank = seed
        self.env_id = env_id
        self.random_reset = random_reset
        self.terminate_on_success = terminate_on_success

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

        # if success, we add "is_success" to the info
        if "success" in info and info["success"]:
            info["is_success"] = True
            if self.terminate_on_success:   #if self.random_reset == "eval":
                done = True
        else:
            info["is_success"] = False

        return obs, reward, done, info

    def get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            observation (object): agent's observation of the current environment
        """
        return self.base_env._get_obs(self.base_env.prev_time_step)
    # @profile
    def reset(self):
        """
        Resets the environment and optionally resets the underlying environment with a random seed.

        Returns:
            observation (object): the initial observation
        """
        self.close()
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
            
        outs = self.base_env.reset()
        import gc
        gc.collect()
        return outs
        # return self.base_env.reset()

    def render(self, mode="rgb_array"):
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
    reward_model,
    image_encoder,
    pca_model=None,
    language_features_policy=None,
    language_features_reward=None,
    use_time=False,
    monitor=False,
    goal_observable=False,
    success_bonus=0.0,
    is_state_based=False,
    mode="train",
    use_proprio=False,
    dense_rewards_at_end=False,
    normalize_reward=False,
    terminate_on_success=False,
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
                env_id, goal_observable=goal_observable, random_reset="eval", terminate_on_success=True
            )
        elif mode == "train":
            base_env = MetaworldBase(
                env_id, goal_observable=goal_observable, random_reset="train", terminate_on_success=terminate_on_success
            )
        elif mode == "demo":
            base_env = MetaworldBase(
                env_id, goal_observable=goal_observable, random_reset="demo", terminate_on_success=True
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


        if reward_model.name == "VLCRewardModel" or reward_model.name == "GVLRewardModel":
            print("language_features_reward", language_features_reward)
            base_env = VLC_GVL_RewardWrapper(
                base_env,
                reward_model,
                image_encoder,
                is_state_based=is_state_based,
                language_features_reward=language_features_reward,
                use_proprio=use_proprio,
                dense_eval=dense_eval,
            )
        else:
            base_env = LearnedRewardWrapper(
                base_env,
                reward_model,
                image_encoder,
                is_state_based=is_state_based,
                language_features_reward=language_features_reward,
                dense_eval=dense_eval,
                use_proprio=use_proprio,
            )

        # This adds the language features to the observation
        if language_features_policy is not None:
            base_env = LanguageWrapper(base_env, language_features_policy)

        # Environment keeps an aggregate reward at each step and outputs it only when the episode ends
        if dense_rewards_at_end:
            base_env = RewardAtEndWrapper(base_env)
        
        if normalize_reward and mode == "train":
            base_env = RewardNormalize(base_env, epsilon=1e-8)
            # base_env = NormalizeReward(base_env, epsilon=1e-8)
            base_env = RecordRewardWrapper(base_env, reward_model)

        # else:
        #     # Then we are an EnvRewardModel
        #     if reward_model.name == 'sparse':
        #         use_sparse = True
        #     elif reward_model.name == 'dense':
        #         use_sparse = False
        #     base_env = RewardWrapper(base_env, sparse=use_sparse, success_bonus=reward_model.success_bonus)

        if monitor:
            base_env = Monitor(base_env)

        return base_env

    return _init


if __name__ == "__main__":
    env = MetaworldBase("door-open-v2", goal_observable=True)
    env.reset()
    env.render()
