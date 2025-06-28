from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
import h5py
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces
import os

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from models.reward_model.base_reward_model import BaseRewardModel

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class CombinedBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    mc_returns: th.Tensor
    offline_data_mask: th.Tensor


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

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    mc_returns: np.ndarray
    offline_data_mask: np.ndarray

    def __init__(
        self,
        h5_path: str,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        success_bonus: float = 0.0,
        add_timestep: bool = False,
        use_language_embeddings: bool = True,
        calculate_mc_returns: bool = False,
        mc_return_gamma: float = 0.99,
        clip_actions: bool = True,
        sparsify_rewards: bool = False,
        dense_rewards_at_end: bool = False,
        filter_instructions: List[str] = None,
        image_encoder: BaseRewardModel = None,
        is_state_based: bool = False,
        use_proprio: bool = False,
        reward_divisor: float = 1.0,
    ):
        """
        Initialize the replay buffer.

        :param h5_path: Path to the HDF5 file that stores the transitions
        :param device: PyTorch device to store the transitions
        :param n_envs: Number of parallel environments
        :param success_bonus: Success bonus added to the rewards
        :param add_timestep: Add a column with the timesteps to the transitions
        :param use_language_embeddings: Whether to specifically incorporate language embeddings into the observations
        :param calculate_mc_returns: Whether to calculate the Monte-Carlo returns
        :param mc_return_gamma: The discount factor for the Monte-Carlo returns
        :param clip_actions: Whether to clip the actions to the action space to [-1, 1]
        :param sparsify_rewards: Converts reward to done
        :param dense_rewards_at_end: Whether to use the reward sum at the end of the episode instead.
        """
        assert not (
            dense_rewards_at_end and sparsify_rewards
        ), "Cannot use both dense rewards at end and sparsify as a precaution"

        print(f"Loading transitions from {h5_path}")
        images = None
        with h5py.File(h5_path, "r") as f:
            observations = f["state"][()]
            lang_embeddings = f["policy_lang_embedding"][()]
            next_observations = observations
            actions = f["action"][()]

            # if 'img' in f.keys() and image_encoder is not None:
            # images = f["img"][()]

            if clip_actions:
                actions = np.clip(actions, -1, 1)
            if sparsify_rewards:
                rewards = f["done"][()]
            else:
                rewards = f["rewards"][()]
            dones = f["done"][()]
            # timesteps = f["timesteps"][()]

            self.is_state_based = is_state_based
            # Process and save images if they are going to be used
            # if not self.is_state_based:
            # image_encoder_preprocessed_path = h5_path.replace(
            #     ".h5", f"_{image_encoder.name}_preprocessed.h5"
            # )
            # # replace "updated_trajs" with "image_encoder_preprocessed"
            # image_encoder_preprocessed_path = (
            #     image_encoder_preprocessed_path.replace(
            #         "updated_trajs", "image_encoder_preprocessed"
            #     )
            # )

            # # Check if the preprocessed file exists
            # try:
            #     with h5py.File(image_encoder_preprocessed_path, "r") as image_f:
            #         encoded = image_f["encoded"][()]

            #     print(
            #         f"Found preprocessed images for {image_encoder.name} in {image_encoder_preprocessed_path}"
            #     )
            # except:
            #     # If not, pre-process the images and save them
            #     images = f["img"]  # Lazy loading with h5py
            #     encoded = image_encoder.encode_images(images)
            #     # create folder if it doesn't exist
            #     os.makedirs(
            #         os.path.dirname(image_encoder_preprocessed_path), exist_ok=True
            #     )
            #     with h5py.File(image_encoder_preprocessed_path, "w") as image_f:
            #         image_f.create_dataset("encoded", data=encoded)

            #     print(
            #         f"Saved preprocessed images for {image_encoder.name} in {image_encoder_preprocessed_path}"
            #     )

            if not self.is_state_based:
                image_encodings = f["img_embedding"][()]
                # If we're using images, let's replace the observations with the encoded
                # proprio is the first 4 observations
                proprio = observations[:, :4]

                img_obs = image_encodings
                if use_proprio:
                    img_obs = np.concatenate((image_encodings, proprio), axis=1)

                observations = img_obs
                next_observations = img_obs

            if filter_instructions is not None:
                instructions = f["env_id"][()]
                indices_to_keep = []
                for i in range(len(instructions)):
                    if instructions[i].decode("utf-8") in filter_instructions:
                        indices_to_keep.append(i)
                observations = observations[indices_to_keep]
                lang_embeddings = lang_embeddings[indices_to_keep]
                next_observations = next_observations[indices_to_keep]
                actions = actions[indices_to_keep]
                rewards = rewards[indices_to_keep]
                dones = dones[indices_to_keep]
            else:
                indices_to_keep = np.arange(observations.shape[0])

            self.indices_to_keep = np.array(indices_to_keep, dtype=int)

        # Use the reward divisor
        # print(f"Using reward divisor: {reward_divisor}", rewards)
        # import pdb ; pdb.set_trace()
        rewards /= reward_divisor

        if dense_rewards_at_end:
            new_rewards = np.zeros_like(rewards)
            prev_start = 0
            for i in range(len(rewards)):
                if dones[i] == 1:
                    new_rewards[i] = np.sum(rewards[prev_start:i])
                    prev_start = i

            rewards = new_rewards
        # add the success bonus
        if success_bonus != 0:
            print(
                "-----Adding success bonus to offline buffer. Warning: this assumes all dones in the offline buffer == success.-----"
            )
            rewards[dones == 1] += success_bonus

        # calculate monte-carlo returns
        self.mc_returns = None
        if calculate_mc_returns:
            # calculate discounted return-to-go for each timestep by using rewards and done
            mc_returns = np.zeros_like(rewards)
            prev_return = 0
            for i in range(len(rewards)):
                mc_returns[-i - 1] = rewards[-i - 1] + mc_return_gamma * prev_return * (
                    1 - dones[-i - 1]
                )
                prev_return = mc_returns[-i - 1]
            self.mc_returns = mc_returns

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

        self.optimize_memory_usage = False

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions.astype(np.float32)
        self.rewards = rewards.astype(np.float32)
        self.dones = dones
        self.timesteps = timesteps
        self.lang_embeddings = np.squeeze(lang_embeddings)

        self.buffer_size = self.rewards.shape[0]
        # self.buffer_size = len(self.indices_to_keep)
        self.success_bonus = success_bonus

        self.pos = self.buffer_size
        self.full = True
        self.device = get_device(device)

        self.add_timestep = add_timestep
        self.use_language_embeddings = use_language_embeddings
        self.calculate_mc_returns = calculate_mc_returns

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
    ) -> CombinedBufferSamples:
        # Batch inds are in sampling indices_to_sample. Get the actual indices
        # batch_inds = np.array([self.indices_to_keep[i] for i in batch_inds])

        # Sample randomly the env idx
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, :],
                env=None,
            )
            # add timestep into the observation
            if self.add_timestep:
                timesteps = (
                    self.timesteps[(batch_inds + 1) % self.buffer_size] / 500
                )  # 500 is the max episode length
                next_obs = np.concatenate((next_obs, timesteps.reshape(-1, 1)), axis=1)

            if self.use_language_embeddings:
                next_obs = np.concatenate(
                    (
                        next_obs,
                        self.lang_embeddings[(batch_inds + 1) % self.buffer_size, :],
                    ),
                    axis=1,
                )

        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, :], env=None
            )
            if self.add_timestep:
                timesteps = (
                    self.timesteps[batch_inds] / 500
                )  # 500 is the max episode length
                next_obs = np.concatenate((next_obs, timesteps.reshape(-1, 1)), axis=1)

            if self.use_language_embeddings:
                next_obs = np.concatenate(
                    (next_obs, self.lang_embeddings[batch_inds, :]), axis=1
                )

        observation = self._normalize_obs(self.observations[batch_inds, :], env=None)

        # add the timestep into the observation
        if self.add_timestep:
            timesteps = (
                self.timesteps[batch_inds] / 500
            )  # 500 is the max episode length
            observation = np.concatenate(
                (observation, timesteps.reshape(-1, 1)), axis=1
            )

        if self.use_language_embeddings:
            observation = np.concatenate(
                (observation, self.lang_embeddings[batch_inds, :]), axis=1
            )

        # set dtype of observations to float32
        observation = observation.astype(np.float32)
        next_obs = next_obs.astype(np.float32)

        rewards = self.rewards[batch_inds].reshape(-1, 1)

        # # set rewards to have all zeros
        # rewards = np.zeros_like(rewards)
        data = (
            observation,
            self.actions[batch_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds].reshape(-1, 1),
            rewards,
            self.mc_returns[batch_inds].reshape(-1, 1)
            if self.calculate_mc_returns
            else rewards,
            np.ones_like(rewards),  # offline_data_mask is 1 for all offline data,
        )

        return CombinedBufferSamples(*tuple(map(self.to_torch, data)))
    
    def _validate_dataset(self, verbose: bool = False) -> None:
        """
        Sanity-check the loaded replay buffer. 不会修改任何数据；
        如发现严重问题直接 raise AssertionError。
        """
        # 1) basic shape/dtype
        assert self.observations.dtype == np.float32, "observations should be float32"
        assert self.next_observations.dtype == np.float32
        assert self.actions.dtype == np.float32
        assert self.rewards.dtype == np.float32
        assert self.dones.dtype in (np.float32, np.float64, np.int8, np.bool_), (
            f"illegal dtype: {self.dones.dtype}"
        )

        # 2) numerical validity
        for name, arr in [
            ("obs", self.observations),
            ("next_obs", self.next_observations),
            ("actions", self.actions),
            ("rewards", self.rewards),
        ]:
            assert np.isfinite(arr).all(), f"{name} contains NaN/Inf"

        # 3) action range
        act_min, act_max = self.actions.min(), self.actions.max()
        assert act_min >= -1.01 and act_max <= 1.01, (
            f"actions out of range: min {act_min:.3f}, max {act_max:.3f}"
        )

        # 4) reward distribution
        r_min, r_max = self.rewards.min(), self.rewards.max()
        # use 1, 200 as empirical thresholds, can be adjusted as needed
        if r_max > 220 or r_min < -220:
            raise AssertionError(f"reward absolute value exceeds 220, likely not scaled correctly: [{r_min}, {r_max}]")

        # 5) success rate and average length
        episode_ends = np.where(self.dones == 1)[0]
        if len(episode_ends) == 0:
            raise AssertionError("dones does not contain 1, cannot identify episode boundaries")

        episode_lengths = np.diff(np.concatenate([[-1], episode_ends])).astype(int)
        avg_H = episode_lengths.mean()
        success_rate = (self.rewards[episode_ends] > 0).mean()

        # 6) expected MC-return (quick estimate)
        gamma = 0.99
        est_return = (gamma ** (avg_H - 1)) * 200 * success_rate + (
            self.rewards[self.dones == 0].mean() * avg_H
        )

        if verbose:
            print("── Data summary ─────────────────────────")
            print(f"buffer size          : {self.buffer_size}")
            print(f"obs dim              : {self.observations.shape[1]}")
            print(f"action dim           : {self.actions.shape[1]}")
            print(f"avg episode length   : {avg_H:.1f}")
            print(f"success rate         : {success_rate*100:.1f}%")
            print(f"reward range         : [{r_min:.3f}, {r_max:.3f}]")
            print(f"estimate MC-return   : {est_return:.2f}")
            print("──────────────────────────────────────────")



class CombinedBuffer(ReplayBuffer):
    def __init__(
        self, old_buffer: ReplayBuffer, new_buffer: ReplayBuffer, ratio: float = 0.5
    ):
        self.old_buffer = old_buffer
        self.new_buffer = new_buffer
        self.ratio = ratio

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> ReplayBufferSamples:
        return

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Add to new buffer
        self.new_buffer.add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        old_batch_size = int(batch_size * self.ratio)
        new_batch_size = batch_size - old_batch_size

        old_samples = self.old_buffer.sample(old_batch_size, env=env)
        new_samples = self.new_buffer.sample(new_batch_size, env=env)
        # Concatenate the samples into old_samples
        cat_names = [
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
            "mc_returns",
            "offline_data_mask",
        ]
        attributes = {}
        for name in cat_names:
            if name == "offline_data_mask":
                # 1 for the old data, 0 for the new data
                old_data = th.ones(old_batch_size, 1)
                new_data = th.zeros(new_batch_size, 1)
            elif name == "mc_returns":
                old_data = getattr(old_samples, name)
                new_data = th.zeros_like(
                    old_data
                )  # set all mc_returns to 0 for new data as it's currently not supported
            else:
                old_data = getattr(old_samples, name)
                new_data = getattr(new_samples, name)
            attributes[name] = th.cat((old_data, new_data), dim=0)

        old_samples = CombinedBufferSamples(**attributes)
        return old_samples

    def size(self) -> int:
        """
        :return: The total size of the buffer
        """
        return self.new_buffer.size() + self.old_buffer.size()


if __name__ == "__main__":
    # Test the H5ReplayBuffer
    h5_path = "/home/yusenluo/RoboCLIP_offline/RoboCLIPv2/scripts/metaworld_policy_pretrain_dataset_rewind_dense.h5"
    buffer = H5ReplayBuffer(h5_path, success_bonus=64)
    buffer._validate_dataset(verbose=True)
    # print(buffer.size())
    # samples = buffer.sample(10)

    # # Test the CombinedBuffer
    # buffer = CombinedBuffer(buffer, buffer)
    # print(buffer.size())
    # samples = buffer.sample(10)

