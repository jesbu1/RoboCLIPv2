from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
import h5py
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

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
        add_timestep: bool = True,
        use_language_embeddings: bool = True
    ):
        """
        Initialize the replay buffer.

        :param h5_path: Path to the HDF5 file that stores the transitions
        :param device: PyTorch device to store the transitions
        :param n_envs: Number of parallel environments
        :param success_bonus: Success bonus added to the rewards
        :param add_timestep: Add a column with the timesteps to the transitions
        """
        with h5py.File(h5_path, "r") as f:
            observations = f["state"][()]
            lang_embeddings = f["lang_embedding"][()]
            next_observations = observations
            actions = f["action"][()]
            rewards = f["rewards"][()]
            # rewards = f["done"][()]
            dones = f["done"][()]
            # timesteps = f["timesteps"][()]

        # TODO: Temporary, but set timesteps to be going from 0-n until it hits a done of 1
        timesteps = np.zeros_like(rewards)
        current_timestep = 0
        for i in range(len(rewards)):
            if dones[i] == 1:
                timesteps[i] = current_timestep
                current_timestep = 0
            else:
                timesteps[i] = current_timestep
                current_timestep += 1
        
        self.optimize_memory_usage = True

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.timesteps = timesteps
        self.lang_embeddings = np.squeeze(lang_embeddings)

        self.buffer_size = self.rewards.shape[0]
        self.success_bonus = success_bonus

        self.pos = self.buffer_size
        self.full = True
        self.device = get_device(device)

        self.add_timestep = add_timestep
        self.use_language_embeddings = use_language_embeddings

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
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, :],
                env=None,
            )
            # add timestep into the observation
            if self.add_timestep:
                timesteps = self.timesteps[(batch_inds + 1) % self.buffer_size] / 500 # 500 is the max episode length
                next_obs = np.concatenate((next_obs, timesteps.reshape(-1, 1)), axis=1)

            if self.use_language_embeddings:
                next_obs = np.concatenate((next_obs, self.lang_embeddings[(batch_inds + 1) % self.buffer_size, :]), axis=1)
                
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, :], env=None
            )
            if self.add_timestep:
                timesteps = self.timesteps[batch_inds] / 500 # 500 is the max episode length
                next_obs = np.concatenate((next_obs, timesteps.reshape(-1, 1)), axis=1)

            if self.use_language_embeddings:
                next_obs = np.concatenate((next_obs, self.lang_embeddings[batch_inds, :]), axis=1)

        observation = self._normalize_obs(self.observations[batch_inds, :], env=None)

        # add the timestep into the observation
        if self.add_timestep:
            timesteps = self.timesteps[batch_inds] / 500 # 500 is the max episode length
            observation = np.concatenate((observation, timesteps.reshape(-1, 1)), axis=1)

        if self.use_language_embeddings:
            observation = np.concatenate((observation, self.lang_embeddings[batch_inds, :]), axis=1)

        # set dtype of observations to float32
        observation = observation.astype(np.float32)
        next_obs = next_obs.astype(np.float32)

        rewards = self.rewards[batch_inds].reshape(-1, 1)

        if self.success_bonus is not None:
            # Give a positive reward if the environment is solved
            success = np.expand_dims(
                np.array([float(done) for done in self.dones[batch_inds]]), -1
            )
            rewards = rewards + self.success_bonus * success



        data = (
            observation,
            self.actions[batch_inds, :].astype(np.float32),
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds].reshape(-1, 1),
            self._normalize_reward(rewards, env=None).astype(np.float32),
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
        attributes = {}
        for name in cat_names:
            old_data = getattr(old_samples, name)
            new_data = getattr(new_samples, name)

            attributes[name] = th.cat((old_data, new_data), dim=0)
        
        old_samples = ReplayBufferSamples(**attributes)
        return old_samples

    def size(self) -> int:
        """
        :return: The total size of the buffer
        """
        return self.new_buffer.size + self.old_buffer.size

if __name__ == "__main__":
    # Test the H5ReplayBuffer
    h5_path = 'data/h5_buffers/updated_trajs/metaworld_dataset_sparse_only.h5'
    buffer = H5ReplayBuffer(h5_path, success_bonus=10)
    print(buffer.size())
    samples = buffer.sample(10)

    breakpoint()