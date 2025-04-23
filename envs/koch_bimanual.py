import random

import numpy as np
import torch
import gym
from gym import Env
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor

from lerobot.common.utils.utils import (
    init_hydra_config,
    init_logging,
    log_say,
    none_or_int,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from hydra.utils import to_absolute_path

from envs.wrappers import *

import time
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect

from lerobot.common.robot_devices.control_utils import log_control_info

import matplotlib.pyplot as plt


import pybullet as p
import pybullet_data
import cv2


def compute_debug_reward(state):
    # In debug mode, we apply a manual reward function based on the current state
    state = state
    # # Let us set the task to be to approach a specific goal position
    goal_position = [
        90,
        0,
        0,
        0,
        0,
        -180,
        90,
        0,
        0,
        0,
        0,
        -180,
    ]

    goal_position = np.array(goal_position)

    # Reward is L2 distance to the goal position from state
    # reward = -torch.norm(state - goal_position)

    # The positions are rotations of motors, so we want the average degree difference
    difference = np.abs(state - goal_position)
    # Bound the difference to 180 degrees
    difference = np.minimum(difference, 180 - difference)

    reward = -np.sum(difference) / 12
    return reward


class KochBimanualEnv(Env):
    def __init__(
        self,
        robot_path,
        max_episode_steps=500,
        fps=30,
        image_keys=[],
        reward_image_key=None,
        fake_robot=False,
    ):
        self.max_episode_steps = max_episode_steps
        self.fps = fps

        self.image_keys = sorted(image_keys, reverse=False)
        self.reward_image_key = reward_image_key
        self.image_reward_idx = self.image_keys.index(reward_image_key)

        robot_cfg = init_hydra_config(robot_path)

        # action space is of size 12
        self.action_space = gym.spaces.Box(
            low=-180, high=180, shape=(12,), dtype=np.float32
        )
        # Make the observation just the state

        # Must define yourself
        self.observation_space = gym.spaces.Dict(
            {
                "proprio": gym.spaces.Box(
                    low=-1, high=1, shape=(12,), dtype=np.float32
                ),
            }
        )
        for key in image_keys:
            self.observation_space.spaces[key] = gym.spaces.Box(
                low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
            )

        self.fake_robot = fake_robot
        if self.fake_robot:
            return

        # For Hydra purposes, must set absolute dir
        robot_cfg["calibration_dir"] = to_absolute_path(robot_cfg["calibration_dir"])
        self.robot = make_robot(robot_cfg)

        self.robot.connect()

        self.reward_image_key = reward_image_key

        self.current_observation = None

        self.counter = 0
        self.prev_time = time.perf_counter()

        self.ax1 = plt.subplot(1, 2, 1)
        self.im1 = plt.imshow(np.random.rand(480, 640, 3))

    def __getstate__(self):
        """Custom method for pickling - exclude cv2 objects and other unpicklable items"""
        state = self.__dict__.copy()

        # Remove the robot object which may contain cv2.VideoCapture
        if "robot" in state:
            del state["robot"]

        # Remove matplotlib objects
        if "ax1" in state:
            del state["ax1"]
        if "im1" in state:
            del state["im1"]

        # Store a flag to indicate we need to reconnect on unpickling
        state["_needs_reconnect"] = not self.fake_robot

        # breakpoint()

        return state

    def __setstate__(self, state):
        """Custom method for unpickling - restore the environment state"""
        # Check if we need to reconnect
        needs_reconnect = state.pop("_needs_reconnect", False)

        # Restore the state
        self.__dict__.update(state)

        # Recreate matplotlib objects if needed
        if not hasattr(self, "ax1") or self.ax1 is None:
            self.ax1 = plt.subplot(1, 2, 1)
            self.im1 = plt.imshow(np.random.rand(480, 640, 3))

        # Reconnect to the robot if needed
        if needs_reconnect and not self.fake_robot:
            robot_path = (
                "/home/abrar/koch_arms/lerobot/lerobot/configs/robot/koch_bimanual.yaml"
            )
            robot_cfg = init_hydra_config(robot_path)
            robot_cfg["calibration_dir"] = to_absolute_path(
                robot_cfg["calibration_dir"]
            )
            self.robot = make_robot(robot_cfg)
            self.robot.connect()

    def ensure_safe_goal_position(
        self,
        goal_pos: torch.Tensor,
        present_pos: torch.Tensor,
        max_relative_target: float | list[float],
    ):
        # Cap relative action target magnitude for safety.
        diff = goal_pos - present_pos
        max_relative_target = torch.tensor(max_relative_target)
        safe_diff = torch.minimum(diff, max_relative_target)
        safe_diff = torch.maximum(safe_diff, -max_relative_target)

        safe_goal_pos = present_pos + safe_diff

        # make joint 0 and 6 be between -90 and 90
        safe_goal_pos[0] = torch.clamp(safe_goal_pos[0], 10, 170)
        safe_goal_pos[6] = torch.clamp(safe_goal_pos[6], 10, 170)

        # Safe diff for

        # if not torch.allclose(goal_pos, safe_goal_pos):
        #     print(
        #         "Relative goal position magnitude had to be clamped to be safe.\n"
        #         f"  requested relative goal position target: {diff.tolist()}\n"
        #         f"    clamped relative goal position target: {safe_diff.tolist()}\n"
        #     )

        return safe_goal_pos

    def step(self, action):
        start_episode_t = time.perf_counter()

        if self.fake_robot:
            # return fake data
            obs = {}
            obs["proprio"] = torch.zeros(12)
            for key in self.image_keys:
                obs[key] = torch.zeros(480, 640, 3)
            return obs, 0, True, {}

        self.counter += 1
        done = False
        reward = 0  # Let a wrapper handle the reward
        info = {}

        if self.counter >= self.max_episode_steps:
            done = True
            self.counter = 0

        if isinstance(action, np.ndarray):
            action = torch.tensor(action).squeeze(0)
        current_state = self.current_observation["observation.state"]

        safe_action = self.ensure_safe_goal_position(
            goal_pos=action, present_pos=current_state, max_relative_target=3.0
        )

        self.robot.send_action(safe_action)
        dt_s = time.perf_counter() - self.prev_time
        # print(f"Time taken: {dt_s}")

        busy_wait(1 / self.fps - dt_s)
        # busy_wait(dt_s)

        # dt_s = time.perf_counter() - self.prev_time
        # log_control_info(self.robot, dt_s, fps=self.fps)
        self.prev_time = time.perf_counter()

        observation = self.robot.capture_observation()

        for key in self.image_keys:
            observation[key] = observation[key] / 255.0

        self.current_observation = observation

        state = observation["observation.state"]

        obs = {}
        obs["proprio"] = state

        for key in self.image_keys:
            obs[key] = observation[key]

        reward = compute_debug_reward(state.numpy())
        # print(reward)
        # reward = 0

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.fake_robot:
            # Return fake data
            return np.random.rand(480, 640, 3)

        # visualize all images using cv2
        for key in self.image_keys:
            image = self.current_observation[key].cpu().numpy()
            cv2.imshow(key, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        obs = self.current_observation["observation.images.main"]

        # turn into a numpy array
        obs = obs.numpy()

        return obs

    def reset(self):
        # TODO: reset somehow
        print("***" * 10, "RESETTING", "***" * 10)
        # reset_position = [
        #     90,
        #     90,
        #     90,
        #     90,
        #     -180,
        #     30,
        #     90,
        #     90,
        #     90,
        #     90,
        #     -180,
        #     30,
        # ]

        reset_position = [
            90,
            90,
            90,
            90,
            180,
            30,
            90,
            90,
            90,
            90,
            180,
            30,
        ]

        if self.fake_robot:
            # return fake data
            obs = {}
            obs["proprio"] = torch.zeros(12)
            for key in self.image_keys:
                obs[key] = torch.zeros(480, 640, 3)
            return obs

        # breakpoint()
        reset_position = torch.tensor(reset_position)

        # Let us linearly interpolate between the reset position and the current position
        # And slowly move the robot to the reset position

        if not self.current_observation:
            observation = self.robot.capture_observation()

            for key in self.image_keys:
                observation[key] = observation[key] / 255.0
            self.current_observation = observation

        # for i in range(50):
        #     current_state = self.current_observation["observation.state"]
        #     new_state = current_state + (reset_position - current_state) / 50
        #     self.robot.send_action(new_state)
        #     observation = self.robot.capture_observation()
        #     self.current_observation = observation

        #     busy_wait(1 / self.fps)

        self.robot.send_action(reset_position)

        # sleep
        # busy_wait(5)
        busy_wait(1)
        observation = self.robot.capture_observation()

        for key in self.image_keys:
            observation[key] = observation[key] / 255.0

        self.current_observation = observation
        self.prev_time = time.perf_counter()

        obs = {}
        obs["proprio"] = observation["observation.state"]
        for key in self.image_keys:
            obs[key] = observation[key]

        return obs

    # Delete the robot
    def close(self):
        if not self.fake_robot:
            self.robot.disconnect()

    def seed(self, seed=None):
        pass


from inputimeout import inputimeout, TimeoutOccurred

from gym import spaces


# Asks the user to provide the reward at the end of the episode
class SuccessWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        state, orig_reward, done, info = self.env.step(action)
        # reward = 0

        if done:
            while True:
                try:
                    answer = None
                    reward = 0.0
                    # If no response in 5 seconds, then assume 0.0
                    try:
                        prompt = "Type '1' in 5 seconds if it is a success, else it is a failure"
                        answer = inputimeout(prompt, timeout=5)
                    except TimeoutOccurred:
                        answer = 0.0

                    if answer:
                        reward = float(answer)

                except:
                    print("Invalid input. Please enter a valid number.")

                if reward == 1.0:
                    info["success"] = True
                else:
                    info["success"] = False
                break

        # reward += orig_reward
        return state, orig_reward, done, info


# Example usage of the base environment and wrappers
def create_wrapped_env(
    env_id,  # ignored here
    reward_model,
    pca_model=None,
    language_features=None,
    policy_language_features=None,
    use_time=False,
    monitor=False,
    goal_observable=False,
    success_bonus=0.0,
    is_state_based=False,
    mode="train",
    use_proprio=True,  # this flag is not used
    dense_rewards_at_end=False,
    action_chunk_size=1,
    camera_kwargs=None,
    robot_disabled=False,
    max_episode_steps=500,
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
        if camera_kwargs is not None:
            image_keys = camera_kwargs["image_keys"]
            reward_image_key = camera_kwargs["reward_image_key"]

        base_env = KochBimanualEnv(
            "/home/abrar/koch_arms/lerobot/lerobot/configs/robot/koch_bimanual.yaml",
            image_keys=image_keys,
            reward_image_key=reward_image_key,
            fake_robot=robot_disabled,
            max_episode_steps=max_episode_steps,
        )

        if pca_model is not None:
            base_env = PCAReducerWrapper(base_env, pca_model)

        if use_time:
            base_env = TimeWrapper(base_env)

        # This replaces the metaworld state-based input with an image embedding too

        base_env = ImageEmbeddingWrapper(base_env, reward_model)

        if not robot_disabled:
            base_env = SuccessWrapper(base_env)

        dense_eval = True if (mode == "eval" or mode == "demo") else False
        base_env = LearnedRewardWrapper(
            base_env,
            reward_model,
            is_state_based=is_state_based,
            language_features=language_features,
            dense_eval=dense_eval,
        )

        # This is all for koch, so this is fine.
        # if reward_model.name == "sparse":

        # This adds the language features to the observation
        if policy_language_features is not None:
            base_env = LanguageWrapper(base_env, policy_language_features)

        # Environment keeps an aggregate reward at each step and outputs it only when the episode ends
        if dense_rewards_at_end:
            base_env = RewardAtEndWrapper(base_env)

        base_env = FlattenDictObservationWrapper(base_env, use_proprio=use_proprio)

        if action_chunk_size > 1:
            if mode == "train":
                base_env = ActionChunkingWrapper(
                    base_env,
                    chunk_size=action_chunk_size,
                    n_action_steps=action_chunk_size,
                )
                # base_env = ACTTemporalEnsemblerWrapper(
                #     base_env, 0.01, action_chunk_size
                # )
            elif mode == "eval" or mode == "demo":
                base_env = ACTTemporalEnsemblerWrapper(
                    base_env, 0.01, action_chunk_size
                )

        if monitor:
            base_env = Monitor(base_env)

        base_env = LoggingWrapper(base_env, logger, prefix=mode)

        return base_env

    return _init


if __name__ == "__main__":
    env = KochBimanualEnv(
        "/home/abrar/koch_arms/lerobot/lerobot/configs/robot/koch_bimanual.yaml"
    )
    obs = env.reset()

    # start = time.time()
    # while viewer.is_running():
    # step_start = time.time()
    # viewer.sync()
    # robot_pos = obs[:6]
    # # convert to radians
    # robot_pos = robot_pos - torch.tensor(offsets)
    # robot_pos = robot_pos * np.pi / 180.0
    # r.set_target_pos(robot_pos)

    for _ in range(6):
        i = _
        print("Tuning joint ", i)
        obs = env.reset()

        action = torch.zeros(12)
        action[i] = -1.0
        action[i + 6] = 1.0

        action *= 0.005

        print(action.tolist())

        for j in range(500):
            # robot_pos = obs[:6]
            # # convert to radians
            # robot_pos = robot_pos - torch.tensor(offsets)
            # robot_pos = robot_pos * np.pi / 180.0
            # r.set_target_pos(robot_pos)
            # viewer.sync()
            # Take zero actions
            print("obs+action", (obs + action).tolist())
            obs, reward, done, info = env.step((action))
            # env.render()
