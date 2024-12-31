from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from offline_rl_algorithms.custom_policies import (
    CustomActor,
    CustomSACPolicy,
    CustomCnnPolicy,
    CustomMlpPolicy,
    CustomMultiInputPolicy,
)


class BC(OfflineRLAlgorithm):
    """
    Behavior Cloning

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
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
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
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": CustomMlpPolicy,
        "CnnPolicy": CustomCnnPolicy,
        "MultiInputPolicy": CustomMultiInputPolicy,
    }
    policy: CustomSACPolicy
    actor: CustomActor

    def __init__(
        self,
        policy: Union[str, Type[CustomSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,  # just to keep it consistent with other offline RL algs
        gamma: float = 0.99,  # just to keep it consistent with other offline RL algs
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1,
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
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.target_update_interval = target_update_interval
        self.name = "bc"

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor

    def train(
        self, gradient_steps: int, batch_size: int = 64, logging_prefix: str = ""
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        actor_losses = []
        actor_log_pis = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Policy loss
            # _, log_prob = self.actor.action_log_prob(replay_data.observations)
            # log_prob = log_prob.reshape(-1, 1)
            # policy_loss = -th.mean(log_prob)

            mean_actions, log_std, kwargs = self.actor.get_action_dist_params(
                replay_data.observations
            )
            distribution = self.actor.action_dist.proba_distribution(
                mean_actions, log_std
            )
            log_prob = distribution.log_prob(replay_data.actions)
            policy_loss = -log_prob.mean()

            # Optimize the policy
            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()

            # log actor stuff
            actor_losses.append(policy_loss.item())
            actor_log_pis.append(log_prob.mean().item())

            # print(f"Actor loss: {policy_loss.item()}")

        self._n_updates += gradient_steps
        # print(np.mean(actor_losses))
        metrics_dict = {
            f"{logging_prefix}/actor_loss": np.mean(actor_losses),
            f"{logging_prefix}/average_reward": replay_data.rewards.mean().item,
            f"{logging_prefix}/average_actor_log_pis": np.mean(actor_log_pis),
        }

        for metric in metrics_dict:
            self.logger.record(logging_prefix + metric, metrics_dict[metric])

        return metrics_dict

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "actor",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = [
            "policy",
        ]
        saved_pytorch_variables = []
        return state_dicts, saved_pytorch_variables
