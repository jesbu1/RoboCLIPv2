from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union
import numpy as np

import torch as th
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.logger import Logger

from offline_rl_algorithms.custom_policies import (
    CustomActor,
    CustomSACPolicy,
    CustomContinuousCritic,
    CustomCnnPolicy,
    CustomMlpPolicy,
    CustomMultiInputPolicy,
)

from offline_rl_algorithms.offline_replay_buffers import CombinedBuffer


class OfflineRLAlgorithm(OffPolicyAlgorithm):
    """
    Base OfflineRL Algorithm Class

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param support_multi_env: Whether to support training with multiple environments
    :param mix_offline_online_buffers: If true, the online replay buffer used during `learn` will be a combination of the offline (from `learn_offline`) and online buffers.
    :param warm_start_online_rl: If true, the online RL training will be warm started with the offline trained policy.
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": CustomMlpPolicy,
        "CnnPolicy": CustomCnnPolicy,
        "MultiInputPolicy": CustomMultiInputPolicy,
    }
    policy: CustomSACPolicy
    actor: CustomActor
    critic: CustomContinuousCritic
    critic_target: CustomContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[CustomSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[spaces.Space]] = (spaces.Box,),
        support_multi_env: bool = True,
        mix_offline_online_buffers: bool = True,
        warm_start_online_rl: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=support_multi_env,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.mix_offline_online_buffers = mix_offline_online_buffers

        self.warm_start_online_rl = warm_start_online_rl
        self.learned_offline = False

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        raise NotImplementedError

    def learn_offline(
        self,
        train_steps: int,
        offline_replay_buffer: ReplayBuffer,
        batch_size: int = 64,
        callback: MaybeCallback = None,
    ) -> None:
        # Getting callbacks to work
        # Create eval callback if needed
        # total_timesteps = 0

        if "current_critic_update_ratio" in self.__dict__:
            # switch to offline
            self.critic_update_ratio = self.offline_critic_update_ratio

        total_timesteps, callback = self._setup_learn(
            total_timesteps=train_steps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="offline",
            progress_bar=False,
        )
        total_timesteps *= self.n_envs  # because of a progress bar issue

        callback = self._init_callback(callback, True)
        callback.on_training_start(locals(), globals())

        # Swap replay buffer for offline training
        old_replay_buffer = self.replay_buffer
        self.replay_buffer = offline_replay_buffer

        print("learning offline")
        self.learned_offline = True
        for _ in range(train_steps):
            metrics = self.train(1, batch_size=batch_size, logging_prefix="offline")
            # metrics is a local() which will be updated in callback.update_locals
            callback.update_locals(locals())  # a little hacky
            callback.on_step()  # because of locals, we have access to self.locals['metrics']

        callback.on_training_end()

        # Set the replay buffer back to the original one
        if self.mix_offline_online_buffers:
            # make a new combined replay buffer with partial sampling of both old and new data
            # for online RL learning
            self.replay_buffer = CombinedBuffer(
                old_buffer=offline_replay_buffer, new_buffer=old_replay_buffer
            )
        else:
            self.replay_buffer = old_replay_buffer

    def train(
        self, gradient_steps: int, batch_size: int = 64, callback: MaybeCallback = None
    ) -> None:
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "OfflineRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        logger: Optional[Logger] = None,
    ):
        if logger is not None:
            super().set_logger(logger)

        if "current_critic_update_ratio" in self.__dict__:
            # switch from offline to online
            self.critic_update_ratio = self.online_critic_update_ratio

        # TODO: implement custom buffer and switch it here
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # def _excluded_save_params(self) -> List[str]:
    #     raise NotImplementedError

    # def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
    #     raise NotImplementedError

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This differs from the parent class in that if there are any offline training steps performed, we will warm start the online RL training with the pre-trained policy.

        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if (
            self.num_timesteps < learning_starts
            and not (self.use_sde and self.use_sde_at_warmup)
            and not (self.warm_start_online_rl and self.learned_offline)
        ):
            # Warmup phase
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
