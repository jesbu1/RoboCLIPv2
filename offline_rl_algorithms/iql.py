from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
from torch import nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import (
    Actor,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    SACPolicy,
)


class ValueCritic(BaseModel):
    """
    Single Value network (state conditioned) for IQL.

    :param observation_space: Obervation space
    :param action_space: Action space (not used, only for compatibility)
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        v_net_list = create_mlp(features_dim, 1, net_arch, activation_fn)
        v_net = nn.Sequential(*v_net_list)
        self.add_module(f"vf", v_net)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        value_input = th.cat([features], dim=1)
        return self.vf(value_input)

class IQL(OfflineRLAlgorithm):
    """
    IQL https://arxiv.org/abs/2110.06169

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
    :param advantage_temp: IQL's advantage temperature
    :param expectile: IQL's expectile regression value
    :param clip_score: Clipping term on the advantage temp
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    q_nets: ContinuousCritic
    q_targets: ContinuousCritic
    v_net: ValueCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
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
        advantage_temp: float = 5.0,
        expectile: float = 0.7,
        clip_score: float = 100,
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

        if _init_setup_model:
            self._setup_model()

        self.advantage_temp = advantage_temp
        self.expectile = expectile
        self.clip_score = clip_score

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic.to(th.float32)
        self.critic_target = self.policy.critic_target

    def train(
        self, gradient_steps: int, batch_size: int = 64, logging_prefix: str = ""
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        q1_values, q2_values = [], []
        q1_next_values, q2_next_values = [], []
        reward_values = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Train the Q Function
            q1_pred, q2_pred = self.critic(replay_data.observations, replay_data.actions)
            target_vf_pred = self.

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            q1_current_actions, q2_current_actions = self.critic(
                replay_data.observations, replay_data.actions
            )

            # CQL Implementation
            random_actions = (
                th.FloatTensor(batch_size, self.action_space.shape[0])
                .uniform_(-1, 1)
                .to(self.device)
            )
            # Sample policy actions to calculate the CQL loss
            current_actions, current_log_pis = self.actor.action_log_prob(
                replay_data.observations
            )
            next_actions, next_log_pis = self.actor.action_log_prob(
                replay_data.next_observations
            )

            # Compute the Q values of random actions
            q1_rand, q2_rand = self.critic(
                replay_data.observations, random_actions.to(th.float32)
            )
            q1_current_actions, q2_current_actions = self.critic(
                replay_data.observations, current_actions.to(th.float32)
            )
            q1_next_actions, q2_next_actions = self.critic(
                replay_data.observations, next_actions.to(th.float32)
            )

            # log q1 and q2 values
            q1_values.append(q1_current_actions.mean().item())
            q2_values.append(q2_current_actions.mean().item())

            # log next q1 and q2 values
            q1_next_values.append(q1_next_actions.mean().item())
            q2_next_values.append(q2_next_actions.mean().item())

            # log average in batch reward
            reward_values.append(replay_data.rewards.mean().item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        metrics_dict = {
            f"{logging_prefix}/actor_loss": np.mean(actor_losses),
            f"{logging_prefix}/critic_loss": np.mean(critic_losses),
            f"{logging_prefix}/average_q1_values": q1_current_actions.mean().item(),
            f"{logging_prefix}/average_q2_values": q2_current_actions.mean().item(),
            f"{logging_prefix}/average_q1_next_values": q1_next_actions.mean().item(),
            f"{logging_prefix}/average_q2_next_values": q2_next_actions.mean().item(),
            f"{logging_prefix}/average_reward": replay_data.rewards.mean().item,
        }

        for metric in metrics_dict:
            self.logger.record(logging_prefix + metric, metrics_dict[metric])

        return metrics_dict

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "IQL",
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
            "critic",
            "critic_target",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = []
        return state_dicts, saved_pytorch_variables
