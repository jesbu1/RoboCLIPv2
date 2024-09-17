from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
import h5py
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class H5ReplayBuffer(ReplayBuffer):
    """
    Replay buffer that can create an HDF5 dataset to store the transitions.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """
    # TODO: success bonus needs to be handled in this replay buffer as an optional param

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        h5_path: str,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        success_bonus: float = 0.0,
    ):
        super().__init__(0, device, n_envs=n_envs)
        self.optimize_memory_usage = True
        with h5py.File(h5_path, "r") as f:
            self.observations = f["state"][()]
            # self.lang_embeddings = f["lang_embedding"][()]
            self.next_observations = self.observations
            self.actions = f["action"][()]
            self.rewards = f["reward"][()]
            self.dones = f["done"][()]
        self.buffer_size = self.actions.shape[0]
        self.success_bonus = success_bonus

    def add(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise (NotImplementedError, "We cannot add transitions to an H5ReplayBuffer")

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.buffer_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, :],
                env=None,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, :], env=None
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, :], env=None),
            self.actions[batch_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds].reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds].reshape(-1, 1), env=None),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class CombinedBuffer(BaseBuffer):
    def __init__(self, old_buffer: ReplayBuffer, new_buffer: ReplayBuffer):
        self.old_buffer = old_buffer
        self.new_buffer = new_buffer

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> ReplayBufferSamples:
        return

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        old_batch_size = batch_size // 2
        new_batch_size = batch_size - old_batch_size
        old_samples = self.old_buffer.sample(old_batch_size, env=env)
        new_samples = self.new_buffer.sample(new_batch_size, env=env)

        # Concatenate the samples into old_samples
        cat_names = ["observations", "actions", "next_observations", "dones", "rewards"]
        for name in cat_names:
            old_data = getattr(old_samples, name)
            new_data = getattr(new_samples, name)
            setattr(old_samples, name, th.cat((old_data, new_data), dim=0))

        return old_samples

    def size(self) -> int:
        """
        :return: The total size of the buffer
        """
        return self.new_buffer.size + self.old_buffer.size
