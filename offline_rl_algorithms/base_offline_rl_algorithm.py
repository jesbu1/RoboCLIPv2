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
import pathlib, io, functools, warnings
from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info
from stable_baselines3.common.save_util import (
    load_from_zip_file,
)


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """
    Recursive version of getattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test
    :param obj:
    :param attr: Attribute to retrieve
    :return: The attribute
    """

    def _getattr(obj: Any, attr: str) -> Any:
        try:
            return getattr(obj, attr, *args)
        except AttributeError:
            print("recursive_getattr failed on " + attr)
            return None

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursive version of setattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_setattr(MyObject, 'sub_object.name', 'hello')
    :param obj:
    :param attr: Attribute to set
    :param val: New value of the attribute
    """
    pre, _, post = attr.rpartition(".")
    try:
        return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)
    except AttributeError:
        print("recursive_setattr failed on " + attr)
        return None


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

        self.replay_buffer = old_replay_buffer

    def set_combined_buffer(
        self, offline_replay_buffer: ReplayBuffer, ratio=0.5
    ) -> None:
        self.replay_buffer = CombinedBuffer(
            old_buffer=offline_replay_buffer, new_buffer=self.replay_buffer, ratio=ratio
        )

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

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                try:
                    attr.load_state_dict(params[name], strict=exact_match)
                except:
                    print("load_state_dict failed on " + name)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    def load(  # noqa: C901
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        load_torch_params_only: bool = False,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )
        if not load_torch_params_only:
            # Remove stored device information and replace with ours
            if "policy_kwargs" in data:
                if "device" in data["policy_kwargs"]:
                    del data["policy_kwargs"]["device"]
                # backward compatibility, convert to new format
                if (
                    "net_arch" in data["policy_kwargs"]
                    and len(data["policy_kwargs"]["net_arch"]) > 0
                ):
                    saved_net_arch = data["policy_kwargs"]["net_arch"]
                    if isinstance(saved_net_arch, list) and isinstance(
                        saved_net_arch[0], dict
                    ):
                        data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

            if (
                "policy_kwargs" in kwargs
                and kwargs["policy_kwargs"] != data["policy_kwargs"]
            ):
                raise ValueError(
                    f"The specified policy kwargs do not equal the stored policy kwargs."
                    f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
                )

            if "observation_space" not in data or "action_space" not in data:
                raise KeyError(
                    "The observation_space and action_space were not given, can't verify new environments"
                )

            if env is not None:
                # Wrap first if needed
                env = self._wrap_env(env, data["verbose"])
                # Check if given env is valid
                check_for_correct_spaces(
                    env, data["observation_space"], data["action_space"]
                )
                # Discard `_last_obs`, this will force the env to reset before training
                # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
                if force_reset and data is not None:
                    data["_last_obs"] = None
                # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
                if data is not None:
                    data["n_envs"] = env.num_envs
            else:
                # Use stored env, if one exists. If not, continue as is (can be used for predict)
                if "env" in data:
                    env = data["env"]

            # noinspection PyArgumentList
            model = self(  # pytype: disable=not-instantiable,wrong-keyword-args
                policy=data["policy_class"],
                env=env,
                device=device,
                _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
            )

            # load parameters
            model.__dict__.update(data)
            model.__dict__.update(kwargs)
            model._setup_model()
        else:
            model = self
        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=False, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model