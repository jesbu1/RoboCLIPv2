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


class KochBimanualEnv(Env):
    def __init__(self, robot_path, max_episode_steps=500, fps=20):
        self.max_episode_steps = max_episode_steps
        self.fps = fps

        robot_cfg = init_hydra_config(robot_path)

        # For Hydra purposes, must set absolute dir
        robot_cfg["calibration_dir"] = to_absolute_path(robot_cfg["calibration_dir"])
        self.robot = make_robot(robot_cfg)

        self.robot.connect()

        # action space is of size 12
        self.action_space = gym.spaces.Box(
            low=-180, high=180, shape=(12,), dtype=np.float32
        )

        # The observation is a 12-dim vector
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(12,), dtype=np.float32
        )

        self.current_observation = None

        self.counter = 0
        self.prev_time = time.perf_counter()

        self.ax1 = plt.subplot(1, 2, 1)
        self.im1 = plt.imshow(np.random.rand(480, 640, 3))

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

        if not torch.allclose(goal_pos, safe_goal_pos):
            print(
                "Relative goal position magnitude had to be clamped to be safe.\n"
                f"  requested relative goal position target: {diff}\n"
                f"    clamped relative goal position target: {safe_diff}"
            )

        return safe_goal_pos

    def step(self, action):
        start_episode_t = time.perf_counter()

        self.counter += 1
        done = False
        reward = 0  # Let a wrapper handle the reward
        info = {}

        if self.counter >= self.max_episode_steps:
            done = True
            self.counter = 0

        if isinstance(action, np.ndarray):
            action = torch.tensor(action)
        current_state = self.current_observation["observation.state"]

        safe_action = self.ensure_safe_goal_position(
            goal_pos=action, present_pos=current_state, max_relative_target=5.0
        )

        self.robot.send_action(safe_action)

        dt_s = time.perf_counter() - self.prev_time
        print(dt_s, 1 / self.fps - dt_s)
        # busy_wait(dt_s)
        busy_wait(1 / self.fps)
        # busy_wait(1 / self.fps)

        dt_s = time.perf_counter() - self.prev_time
        log_control_info(self.robot, dt_s, fps=self.fps)

        observation = self.robot.capture_observation()

        self.current_observation = observation

        state = observation["observation.state"]

        self.prev_time = time.perf_counter()

        # # # Let us set the task to be to approach a specific goal position
        # goal_position = [
        #     90,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     90,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        # ]

        # goal_position = torch.tensor(goal_position)

        # # Reward is L2 distance to the goal position from state
        # # reward = -torch.norm(state - goal_position)

        # # The positions are rotations of motors, so we want the average degree difference
        # difference = torch.abs(state - goal_position)
        # # Bound the difference to 180 degrees
        # difference = torch.min(difference, 180 - difference)
        # reward = -torch.sum(difference) / 12

        # print(reward)
        # print()

        # # if the difference is less than 2 degrees, we can set success to True
        # if torch.norm(state - goal_position) < 2.0:
        #     info["success"] = True

        return state, reward, done, info

    def render(self, mode="rgb_array"):
        # obs = []
        # for key in self.current_observation:
        #     if "image" in key:
        #         obs.append(self.current_observation[key])
        # Return the image

        # Let's only use 1 camera

        obs = self.current_observation["observation.images.main"]

        # turn into a numpy array
        obs = obs.numpy()

        self.im1.set_data(obs)
        plt.pause(1 / self.fps)

        return obs

    def reset(self):
        # TODO: reset somehow
        print("***" * 10, "RESETTING", "***" * 10)
        reset_position = [
            90,
            90,
            90,
            90,
            0,
            30,
            90,
            90,
            90,
            90,
            0,
            30,
        ]

        # breakpoint()
        reset_position = torch.tensor(reset_position)

        # Let us linearly interpolate between the reset position and the current position
        # And slowly move the robot to the reset position

        if not self.current_observation:
            observation = self.robot.capture_observation()
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
        busy_wait(10)
        observation = self.robot.capture_observation()
        self.current_observation = observation
        self.prev_time = time.perf_counter()

        return observation["observation.state"]

    # Delete the robot
    def close(self):
        self.robot.disconnect()

    def seed(self, seed=None):
        pass

    # def _update_sim(self, state):
    #     # update simulator
    #     obs_left = state[:6]
    #     obs_right = state[6:]

    #     obs_left = obs_left - self.offset
    #     obs_right = obs_right - self.offset

    #     obs_left = obs_left * np.pi / 180.0
    #     obs_right = obs_right * np.pi / 180.0

    #     p.setJointMotorControlArray(
    #         self.robot_id1,
    #         jointIndices=range(len(obs_left)),
    #         controlMode=p.POSITION_CONTROL,
    #         targetPositions=obs_left,
    #     )
    #     p.stepSimulation()

    #     p.setJointMotorControlArray(
    #         self.robot_id2,
    #         jointIndices=range(len(obs_right)),
    #         controlMode=p.POSITION_CONTROL,
    #         targetPositions=obs_right,
    #     )
    #     p.stepSimulation()

    #     link_state1 = p.getLinkState(self.robot_id1, 6, computeForwardKinematics=True)
    #     link_state2 = p.getLinkState(self.robot_id2, 6, computeForwardKinematics=True)

    #     self.current_ee1 = link_state1[0]
    #     self.current_ee2 = link_state2[0]

    #     self.current_ee1_rpy = p.getEulerFromQuaternion(link_state1[1])
    #     self.current_ee2_rpy = p.getEulerFromQuaternion(link_state2[1])

    #     # Conver to torch tensor
    #     self.current_ee1 = torch.tensor(self.current_ee1)
    #     self.current_ee2 = torch.tensor(self.current_ee2)
    #     self.current_ee1_rpy = torch.tensor(self.current_ee1_rpy)
    #     self.current_ee2_rpy = torch.tensor(self.current_ee2_rpy)

    # def _ee_to_state(self, ee1, ee2, ee1_rpy, ee2_rpy):
    #     # pos1 = torch.tensor(self.r1.inverse_kinematics_rot(ee1, ee1_rpy))
    #     # pos2 = torch.tensor(self.r2.inverse_kinematics_rot(ee2, ee2_rpy))

    #     # let's to ik here
    #     pos1 = p.calculateInverseKinematics(self.robot_id1, 6, ee1, ee1_rpy)
    #     pos2 = p.calculateInverseKinematics(self.robot_id2, 6, ee2, ee2_rpy)

    #     pos1 = torch.tensor(pos1)
    #     pos2 = torch.tensor(pos2)
    #     # convert to degrees
    #     pos1 = pos1 * 180.0 / np.pi


from inputimeout import inputimeout, TimeoutOccurred


# Asks the user to provide the reward at the end of the episode
class ManualRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            while True:
                try:
                    reward = 0
                    answer = None
                    # If no response in 5 seconds, then assume 0.0
                    try:
                        prompt = "Give a reward in 2 seconds, else it is a failure"
                        answer = inputimeout(prompt, timeout=5)
                    except TimeoutOccurred:
                        answer = 0.0

                    if answer:
                        reward = float(answer)

                    break
                except:
                    print("Invalid input. Please enter a valid number.")

                if reward == 1.0:
                    info["success"] = True
                else:
                    info["success"] = False
        return state, reward, done, info


# Example usage of the base environment and wrappers
def create_wrapped_env(
    env_id,  # ignored here
    reward_model,
    pca_model=None,
    language_features=None,
    use_time=False,
    monitor=False,
    goal_observable=False,
    success_bonus=0.0,
    is_state_based=False,
    mode="train",
    use_proprio=True,  # this flag is not used
    dense_rewards_at_end=False,
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
        base_env = KochBimanualEnv(
            "/home/abrar/koch_arms/lerobot/lerobot/configs/robot/koch_bimanual.yaml"
        )

        if pca_model is not None:
            base_env = PCAReducerWrapper(base_env, pca_model)

        if use_time:
            base_env = TimeWrapper(base_env)

        # breakpoint()
        # This replaces the metaworld state-based input with an image embedding too

        dense_eval = True if (mode == "eval" or mode == "demo") else False

        base_env = LearnedRewardWrapper(
            base_env,
            reward_model,
            is_state_based=is_state_based,
            language_features=language_features,
            dense_eval=dense_eval,
            use_proprio=use_proprio,
        )

        # This is all for koch, so this is fine.
        if reward_model.name == "sparse" or reward_model.name == "dense":
            base_env = ManualRewardWrapper(base_env)

        # This adds the language features to the observation
        if language_features is not None:
            base_env = LanguageWrapper(base_env, language_features)

        # Environment keeps an aggregate reward at each step and outputs it only when the episode ends
        if dense_rewards_at_end:
            base_env = RewardAtEndWrapper(base_env)

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
