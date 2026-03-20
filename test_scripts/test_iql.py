from gym import Env, spaces
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from offline_rl_algorithms.offline_replay_buffers import H5ReplayBuffer
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
import torch as th
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch as th
import numpy as np
import os
from memory_profiler import profile
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to get rid of the warning message
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from typing import Any, Dict

import torch as th

import argparse
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import metaworld
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)

# from kitchen_env_wrappers import readGif
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import io
import random
import torch.nn.functional as F


from offline_rl_algorithms.cql import CQL
from offline_rl_algorithms.iql import IQL
from offline_rl_algorithms.bc import BC
from offline_rl_algorithms.rlpd import RLPD

from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from offline_rl_algorithms.wandb_logger import WandBLogger
from offline_rl_algorithms.callbacks import CustomWandbCallback, OfflineEvalCallback

# from models.reward_model.xclip_encoder import XCLIPEncoder
# from models.reward_model import image_encoders

from models.reward_model.base_reward_model import BaseRewardModel
from models.reward_model.rewind_reward_model import RewindRewardModel
from models.encoders.dino_miniLM_encoder import Dino_miniLM_Encoder
from models.reward_model.env_reward_model import EnvRewardModel

from envs.metaworld_envs.metaworld import (
    create_wrapped_env,
    instruction_to_environment,
    environment_to_instruction,
)


from stable_baselines3.common.policies import ActorCriticPolicy
import stable_baselines3

from test_scripts.eval_utils import offline_eval
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import EvalCallback

import torch as th


# th.set_float32_matmul_precision("high")
def create_exp_name(cfg: DictConfig):
    exp_name = cfg.environment.cfg_name + "_"

    exp_name += cfg.reward_model.name + "_"

    exp_name += cfg.general_training.name + "_"

    exp_name += "_longer_online"

    if cfg.general_training.algo == "iql":
        # add policy_extraction and awr/ddpg params
        exp_name += f"pe_{cfg.general_training.policy_extraction}_"

        if cfg.general_training.policy_extraction == "awr":
            exp_name += f"adv_temp_{cfg.general_training.awr_advantage_temp}_"
        elif cfg.general_training.policy_extraction == "ddpg":
            exp_name += f"bc_weight_{cfg.general_training.ddpg_bc_weight}_"

        exp_name += f"n_critics_{cfg.general_training.n_critics}_"
        exp_name += f"n_critics_to_sample_{cfg.general_training.n_critics_to_sample}_"
        exp_name += f"utd_{cfg.online_training.critic_update_ratio}_"

    if cfg.general_training.algo == "cql":
        exp_name += f"min_q_weight_{cfg.general_training.cql_min_q_weight}_"
        exp_name += f"min_q_temp_{cfg.general_training.cql_min_q_temp}_"
        exp_name += f"n_critics_{cfg.general_training.n_critics}_"
        exp_name += f"n_critics_to_sample_{cfg.general_training.n_critics_to_sample}_"
        exp_name += f"utd_{cfg.online_training.critic_update_ratio}_"

    if cfg.general_training.algo == "rlpd":
        exp_name += f"n_critics_{cfg.general_training.n_critics}_"
        exp_name += f"n_critics_to_sample_{cfg.general_training.n_critics_to_sample}_"
        exp_name += f"train_critic_with_entropy_{cfg.general_training.rlpd_train_critic_with_entropy}_"
        exp_name += f"utd_{cfg.online_training.critic_update_ratio}_"

    if cfg.environment.ignore_language:
        exp_name += "no_lang_"

    if cfg.environment.is_state_based:
        exp_name += "state_based_"

    if cfg.environment.use_proprio:
        exp_name += "use_proprio_"

    exp_name += f"_seed_{cfg.environment.env_id}"
    exp_name += f"_seed_{cfg.general_training.seed}"

    if cfg.reward_model.reward_at_every_step:
        exp_name += "_dense"
    else:
        exp_name += "_sparse"

    if cfg.general_training.normalize_reward:
        exp_name += "_normalize"

    # if the last character is an underscore, remove it
    if exp_name[-1] == "_":
        exp_name = exp_name[:-1]

    return exp_name


def parse_reward_model(reward_cfg: DictConfig) -> BaseRewardModel:
    reward_string = reward_cfg.name
    if reward_string is None:
        return None
    if reward_string == "roboclip":
        from models.reward_model.roboclip_reward_model import RoboclipRewardModel
        reward_model = RoboclipRewardModel(
            model_load_path=reward_cfg.model_path,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
        )
    elif reward_string == "vlc":
        from models.reward_model.vlc_reward_model import VLCRewardModel
        reward_model = VLCRewardModel(
            server_url=reward_cfg.server_url,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
            reward_at_every_step=reward_cfg.reward_at_every_step,
        )
    elif reward_string == "gvl":
        from models.reward_model.gvl_reward_model import GVLRewardModel
        reward_model = GVLRewardModel(
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
        )
    elif reward_string == "rewind":
        reward_model = RewindRewardModel(
            model_load_path=reward_cfg.model_path,
            use_pca=reward_cfg.use_pca,
            attention_heads=4,
            pca_model_dir=None,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
            sum_reward=reward_cfg.sum_reward,
            reward_at_every_step=reward_cfg.reward_at_every_step,
        )
    elif reward_string == "liv":
        from models.reward_model.liv_reward_model import LIVRewardModel
        reward_model = LIVRewardModel(
            model_load_path=reward_cfg.model_path,
            use_pca=reward_cfg.use_pca,
            attention_heads=4,
            pca_model_dir=None,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
            last_frame_reward_only=reward_cfg.last_frame_reward_only,
            reward_at_every_step=reward_cfg.reward_at_every_step,
        )

    # TODO: get these models up
    elif reward_string == "sparse":
        # raise NotImplementedError("Sparse reward model not implemented yet.")
        reward_model = EnvRewardModel(
            reward_type="sparse",
            model_path=reward_cfg.model_path,
            success_bonus=reward_cfg.success_bonus,
        )
    elif reward_string == "dense":
        reward_model = EnvRewardModel(
            reward_type="dense",
            model_path=reward_cfg.model_path,
            success_bonus=reward_cfg.success_bonus,
        )
    else:
        raise ValueError(f"Unknown reward model: {reward_string}")

    # Set the success bonus
    reward_model.set_success_bonus(reward_cfg.success_bonus)
    reward_model.set_reward_divisor(reward_cfg.reward_divisor)
    print(f"Success bonus: {reward_model.success_bonus}")

    # Also set default image encoder to be a LIVEncoder
    # image_encoder = LIVEncoder(
    #     model_load_path="",
    #     use_pca=False,
    #     attention_heads=4,
    #     device="cuda",
    #     batch_size=64,
    # )
    image_encoder = Dino_miniLM_Encoder(
        use_pca=False,
        device="cuda",
        dino_batch_size=128,
        max_num_frames_per_episode=128,
        batch_size=64,
    )

    return reward_model, image_encoder


# Define the function to initialize Hydra
# @profile
@hydra.main(config_path="../configs", config_name="base_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Extract configurations
    training_config = cfg.general_training
    env_config = cfg.environment
    model_config = cfg.model
    logging_config = cfg.logging
    offline_config = cfg.offline_training

    experiment_name = create_exp_name(cfg)

    ### Setup wandb and logging ###
    if logging_config.wandb:
        config_for_wandb = OmegaConf.to_container(cfg, resolve=True)
        absolute_log_dir = os.path.abspath(logging_config.log_dir)
        print("absolute log dir is", absolute_log_dir)
        config_for_wandb["log_dir"] = absolute_log_dir
        wandb.init(
            entity=logging_config.wandb_entity_name,
            project=logging_config.wandb_project_name,
            group=logging_config.wandb_group_name,
            name=experiment_name,
            config=config_for_wandb,
            monitor_gym=True,
            sync_tensorboard=True,
            notes=cfg.wandb_notes,
        )

    log_dir = logging_config.log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    reward_model, image_encoder = parse_reward_model(cfg.reward_model)

    ### Create environment and callbacks ###
    envs, eval_env = create_envs(cfg, reward_model, image_encoder)
    model, model_class, policy_kwargs = get_policy_algorithm(cfg, envs, log_dir)

    # Set eval freq and video freq if not set
    # if it's rlpd, video_freq should be never
    if training_config.algo == "rlpd":
        # video_freq = 0
        # eval_freq = 0
        eval_freq = offline_config.offline_training_steps * env_config.n_envs // (5)
        video_freq = eval_freq
    else:
        video_freq = offline_config.offline_training_steps * env_config.n_envs // 1
        eval_freq = offline_config.offline_training_steps * env_config.n_envs // (1)

    # Use deterministic actions for evaluation
    eval_callback = OfflineEvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        video_freq=video_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=25,
    )

    callback_list = generate_callback_list(logging_config, eval_callback)

    # Create the logger
    wandb_logger = WandBLogger()
    model.set_logger(wandb_logger)

    use_language = not env_config.ignore_language

    if offline_config.offline_tasks == "all":
        offline_tasks = None
    else:
        offline_tasks = offline_config.offline_tasks

    # Map the tasks to their strings
    # offline_task_strings =
    if (
        offline_config.offline_training_steps > 0
        or cfg.online_training.mix_buffers_ratio > 0.0
    ):
        try:
            h5_path = offline_config.offline_h5_path.format(cfg.reward_model.name)
            h5_path = to_absolute_path(h5_path)
            print(h5_path)
            # check if file exists
            with open(h5_path, "r") as f:
                pass
        except FileNotFoundError:
            print(
                "File {} not found. This file likely does not have the correct reward preprocessed.".format(
                    h5_path
                )
            )
            raise FileNotFoundError

    if (
        offline_config.offline_training_steps > 0
        or cfg.online_training.mix_buffers_ratio > 0.0
    ):
        sparse_only = True if reward_model.name == "sparse" else False
        buffer = H5ReplayBuffer(
            h5_path,
            use_language_embeddings=use_language,
            success_bonus=cfg.reward_model.success_bonus,
            sparsify_rewards=sparse_only,
            filter_instructions=offline_tasks,
            image_encoder=reward_model,
            is_state_based=env_config.is_state_based,
            use_proprio=env_config.use_proprio,
            calculate_mc_returns=training_config.use_calibrated_q,  # only used for cal-ql
            mc_return_gamma=training_config.gamma,
            dense_rewards_at_end=training_config.dense_rewards_at_end,
            reward_divisor=cfg.reward_model.reward_divisor,
        )

    ### Learn offline
    if (
        offline_config.offline_training_steps > 0
        and isinstance(model, OfflineRLAlgorithm)
        or (hasattr(offline_config, "ckpt_path") and offline_config.ckpt_path)
    ):
        # if offline_config.offline_h5_path is None:
        #     default_h5_path = (
        #         "data/h5_buffers/updated_trajs/metaworld_traj_100_demos_orig_reward.h5"
        #     )
        #     print("There is no h5 path provided. Defaulting to", default_h5_path)
        #     h5_path = default_h5_path
        # else:
        #     h5_path = offline_config.offline_h5_path

        if hasattr(offline_config, "ckpt_path") and offline_config.ckpt_path:
            # convert to absolute path from hydra
            offline_config.ckpt_path = to_absolute_path(offline_config.ckpt_path)
            print(f"Loading checkpoint from {offline_config.ckpt_path}")

            # if rlpd, we do a special load
            if training_config.algo == "rlpd":
                print(
                    "Loading offline algo from",
                    offline_config.ckpt_path + "_rlpd_offline",
                )
                # load the original offline algo
                offline_algo = model.offline_algo.load(
                    offline_config.ckpt_path + "_rlpd_offline",
                    env=envs,
                    custom_objects={
                        "observation_space": envs.observation_space,
                        "action_space": envs.action_space,
                    },
                    print_system_info=True,
                    load_torch_params_only=True,  # new argument to make sure config isn't overwritten
                )

                # load the current model
                model = model.load(
                    offline_config.ckpt_path,
                    env=envs,
                    custom_objects={
                        "observation_space": envs.observation_space,
                        "action_space": envs.action_space,
                    },
                    print_system_info=True,
                    load_torch_params_only=True,  # new argument to make sure config isn't overwritten
                )

                model.offline_algo = offline_algo

                model.set_logger(wandb_logger)
                model.learned_offline = True
                model.set_policies_with_offline()

            else:
                model.load(offline_config.ckpt_path, env=envs)
        else:
            model.learn_offline(
                offline_replay_buffer=buffer,
                train_steps=offline_config.offline_training_steps,
                callback=callback_list,
                batch_size=256,
            )

            # save the model in log_dir/last
            save_dir = os.path.join(log_dir, "last_offline")

            if training_config.algo == "rlpd":
                model.offline_algo.save(save_dir + "_rlpd_offline")
                model.save(save_dir, exclude=["offline_algo"])
            else:
                model.save(save_dir)
            # Model is saved at
            absolute_save_dir = os.path.abspath(save_dir)
            print(f"Model saved at {absolute_save_dir}")
            # Log this absolute_save_dir in wandb
            if logging_config.wandb:
                wandb.run.log({"model_dir": absolute_save_dir})
                

    # add eval policy on all tasks
    #offline_eval(model, reward_model, image_encoder)
    # exit()


    # Set the replay buffer back to the original one
    if (
        isinstance(model, OfflineRLAlgorithm)
        and cfg.online_training.mix_buffers_ratio > 0.0
    ):
        model.set_combined_buffer(buffer, ratio=cfg.online_training.mix_buffers_ratio)

    ### Learn online ###
    logger = model.logger  # set logger in case
    online_eval_freq = logging_config.eval_freq // env_config.n_envs  # // args.nenvsto
    online_video_freq = logging_config.video_freq // env_config.n_envs
    eval_callback.eval_freq = online_eval_freq
    eval_callback.video_freq = online_video_freq
    # assert cfg.online_training.total_time_steps == 500000, "Total time steps should be 500k."
    if cfg.online_training.total_time_steps > 0:
        # logger only exists for offline algorithms
        if isinstance(model, OfflineRLAlgorithm):
            model.learn(
                total_timesteps=int(cfg.online_training.total_time_steps),
                callback=callback_list,
                logger=logger,
                progress_bar=True,
            )
        else:
            model.learn(
                total_timesteps=int(cfg.online_training.total_time_steps),
                callback=callback_list,
                progress_bar=True,
            )
    model.save(log_dir)

    if logging_config.wandb:
        wandb.finish()


def create_envs(cfg: DictConfig, reward_model: BaseRewardModel, image_encoder):
    env_config = cfg.environment
    # Extract configuration
    # env_id = env_config.env_id
    # env_id = instruction_to_environment[env_config.text_string]
    # text_instruction = env_config.text_string
    env_id = env_config.env_id
    text_instruction = environment_to_instruction[env_id]

    with th.no_grad():
        lang_feat_policy = reward_model.encode_text_for_policy(text_instruction).squeeze()
        # lang_feat = th.from_numpy(lang_feat).squeeze()
        if reward_model.name != "GVLRewardModel" and reward_model.name != "VLCRewardModel":
            lang_feat_reward = reward_model.encode_text(text_instruction).squeeze()
            print("Lang feat reward shape", lang_feat_reward.shape)
        else:
            lang_feat_reward = text_instruction

    # print("Lang feat policy shape", lang_feat_policy.shape) # 384
    # 
    ignore_language = env_config.ignore_language

    # Define envs (dummy example for illustration)
    # assert env_config.n_envs == 4, "Number of environments should be 4."
    if env_config.n_envs > 1:
        envs = SubprocVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    language_features_policy=lang_feat_policy if not ignore_language else None,
                    language_features_reward=lang_feat_reward,
                    reward_model=reward_model,
                    image_encoder=image_encoder,
                    goal_observable=True,
                    success_bonus=cfg.reward_model.success_bonus,
                    is_state_based=env_config.is_state_based,
                    use_proprio=env_config.use_proprio,
                    mode="train",
                    dense_rewards_at_end=cfg.general_training.dense_rewards_at_end,
                    normalize_reward=cfg.general_training.normalize_reward,
                    terminate_on_success=cfg.general_training.terminate_on_success,
                )
                for _ in range(env_config.n_envs)
            ]
        )

        # envs = LoggingWrapper(envs)
    else:
        envs = DummyVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    success_bonus=cfg.reward_model.success_bonus,
                    language_features_policy=lang_feat_policy if not ignore_language else None,
                    language_features_reward=lang_feat_reward,
                    reward_model=reward_model,
                    image_encoder=image_encoder,
                    goal_observable=True,
                    is_state_based=env_config.is_state_based,
                    use_proprio=env_config.use_proprio,
                    mode="train",
                    dense_rewards_at_end=cfg.general_training.dense_rewards_at_end,
                    normalize_reward=cfg.general_training.normalize_reward,
                    terminate_on_success=cfg.general_training.terminate_on_success,
                )
            ]
        )

    if env_config.n_envs > 1:
        eval_env = SubprocVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    reward_model=reward_model,
                    image_encoder=image_encoder,
                    language_features_policy=lang_feat_policy if not ignore_language else None,
                    language_features_reward=lang_feat_reward,
                    monitor=True,
                    goal_observable=True,
                    is_state_based=env_config.is_state_based,
                    mode="eval",
                    use_proprio=env_config.use_proprio,
                    normalize_reward=cfg.general_training.normalize_reward,
                    terminate_on_success=cfg.general_training.terminate_on_success,
                )
                for i in range(1)
            ]
        )  # KitchenEnvDenseOriginalReward(time=True)
    else:
        eval_env = DummyVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    reward_model=reward_model,
                    image_encoder=image_encoder,
                    language_features_policy=lang_feat_policy if not ignore_language else None,
                    language_features_reward=lang_feat_reward,
                    monitor=True,
                    goal_observable=True,
                    is_state_based=env_config.is_state_based,
                    mode="eval",
                    use_proprio=env_config.use_proprio,
                    normalize_reward=cfg.general_training.normalize_reward,
                    terminate_on_success=cfg.general_training.terminate_on_success,
                )
            ]
        )  # KitchenEnvDenseOriginalReward(time=True)
    return envs, eval_env


def get_policy_algorithm(cfg: DictConfig, envs: VecEnv, log_dir: str):
    env_config = cfg.environment
    model_config = cfg.model

    args = cfg.general_training

    if cfg.general_training.action_noise is not None:
        n_actions = envs.action_space.shape[-1]
        action_noise = stable_baselines3.common.noise.NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=cfg.general_training.action_noise * n_actions,
        )
    else:
        action_noise = None

    if args.algo.lower() == "ppo":
        policy_kwargs = {
            "net_arch": dict(pi=model_config.pi_net_arch, vf=model_config.vf_net_arch),
            "policy_layer_norm": model_config.policy_layer_norm,
            "critic_layer_norm": model_config.critic_layer_norm,
        }
    else:
        policy_kwargs = {
            "net_arch": dict(pi=model_config.pi_net_arch, qf=model_config.qf_net_arch),
            "policy_layer_norm": model_config.policy_layer_norm,
            "critic_layer_norm": model_config.critic_layer_norm,
        }

    # everything except BC, SAC, and PPO require n_critics
    if (
        cfg.general_training.algo == "iql"
        or cfg.general_training.algo == "cql"
        or cfg.general_training.algo == "rlpd"
    ):
        policy_kwargs["n_critics"] = cfg.general_training.n_critics

    algo = args.algo.lower()

    # If using RLPD, we instantiate a different offline model
    orig_algo = algo
    if args.algo.lower() == "rlpd":
        algo = cfg.general_training.rlpd_offline_algo.lower()

    if algo == "ppo":
        model_class = PPO

        policy_kwargs = {
            "net_arch": policy_kwargs["net_arch"],
        }

        if not args.pretrained:
            model = model_class(
                "MlpPolicy",                        # 策略类型，与 SAC 一致
                envs,                              # 环境对象，与 SAC 一致
                verbose=1,                         # 日志详细程度，与 SAC 一致
                tensorboard_log=log_dir,           # TensorBoard 日志目录，与 SAC 一致
                n_steps=args.n_steps,              # PPO 特有，每个更新周期的步数
                batch_size=args.n_steps * env_config.n_envs,  # PPO 特有，批次大小
                n_epochs=10,                        # PPO 特有，训练 epoch 数
                ent_coef=args.entropy_term,        # 熵系数，与 SAC 一致
                learning_rate=args.learning_rate,  # 学习率，与 SAC 一致
                seed=args.seed,                    # 随机种子，与 SAC 一致
                policy_kwargs=policy_kwargs,       # 策略网络参数，与 SAC 一致
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif algo == "sac":
        model_class = SAC

        # For SAC, we cannot take anything besides net_arch as a parameter
        policy_kwargs = {
            "net_arch": policy_kwargs["net_arch"],
        }

        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                # batch_size=args.n_steps * args.n_envs,
                ent_coef=args.entropy_term,
                buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                seed=args.seed,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                train_freq=(
                    cfg.environment.train_freq_num,
                    cfg.environment.train_freq_type,
                ),
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif algo in ["cql"]:
        model_class = CQL
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                ent_coef=args.entropy_term,
                buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                seed=args.seed,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                train_freq=(
                    cfg.environment.train_freq_num,
                    cfg.environment.train_freq_type,
                ),
                online_critic_update_ratio=cfg.online_training.critic_update_ratio,
                offline_critic_update_ratio=cfg.offline_training.critic_update_ratio,
                min_q_weight=cfg.general_training.cql_min_q_weight,
                min_q_temp=cfg.general_training.cql_min_q_temp,
                use_calibrated_q=cfg.general_training.use_calibrated_q,
                n_critics_to_sample=cfg.general_training.n_critics_to_sample,
                warm_start_online_rl=cfg.online_training.warm_start_online_rl,
                gamma=cfg.general_training.gamma,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    elif algo == "iql":
        model_class = IQL

        # policy = SACPolicy(observation_space=envs.observation_space, action_space=envs.action_space, net_arch=[32, 32], lr_schedule=None)
        # if not args.pretrained:
        model = model_class(
            "MlpPolicy",
            envs,
            verbose=1,
            tensorboard_log=log_dir,
            buffer_size=cfg.online_training.total_time_steps,
            learning_starts=cfg.online_training.learning_starts,
            seed=args.seed,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            train_freq=(
                cfg.environment.train_freq_num,
                cfg.environment.train_freq_type,
            ),
            online_critic_update_ratio=cfg.online_training.critic_update_ratio,
            offline_critic_update_ratio=cfg.offline_training.critic_update_ratio,
            policy_extraction=cfg.general_training.policy_extraction,
            advantage_temp=cfg.general_training.awr_advantage_temp,
            ddpg_bc_weight=cfg.general_training.ddpg_bc_weight,
            n_critics_to_sample=cfg.general_training.n_critics_to_sample,
            warm_start_online_rl=cfg.online_training.warm_start_online_rl,
            gamma=cfg.general_training.gamma,
            # expectile=cfg.general_training.expectile,
        )
        if args.pretrained:
            print("Resuming model from", cfg.general_training.resume_ckpt_path)
            model = model.load(
                cfg.general_training.resume_ckpt_path,
                env=envs,
                custom_objects={
                    "observation_space": envs.observation_space,
                    "action_space": envs.action_space,
                },
                print_system_info=True,
                load_torch_params_only=True,  # new argument to make sure config isn't overwritten
            )
    elif algo == "bc":
        model_class = BC
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                seed=args.seed,
                action_noise=action_noise,  # should be null?
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                train_freq=(
                    cfg.environment.train_freq_num,
                    cfg.environment.train_freq_type,
                ),  # useless
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    if orig_algo.lower() == "rlpd":
        offline_model = model
        model_class = RLPD

        # if not args.pretrained:
        model = model_class(
            "MlpPolicy",
            envs,
            offline_algo=model,
            verbose=1,
            tensorboard_log=log_dir,
            buffer_size=cfg.online_training.total_time_steps,
            learning_starts=cfg.online_training.learning_starts,
            seed=args.seed,
            action_noise=action_noise,  # should be null
            ent_coef=args.entropy_term,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            train_freq=(
                cfg.environment.train_freq_num,
                cfg.environment.train_freq_type,
            ),  # useless
            online_critic_update_ratio=cfg.online_training.critic_update_ratio,
            offline_critic_update_ratio=cfg.offline_training.critic_update_ratio,
            n_critics_to_sample=cfg.general_training.n_critics_to_sample,
            train_critic_with_entropy=cfg.general_training.rlpd_train_critic_with_entropy,
            warm_start_online_rl=cfg.online_training.warm_start_online_rl,
            gamma=cfg.general_training.gamma,
        )
        # load the current model
        if args.pretrained:
            print("Resuming model from", cfg.general_training.resume_ckpt_path)
            model = model.load(
                cfg.general_training.resume_ckpt_path,
                env=envs,
                custom_objects={
                    "observation_space": envs.observation_space,
                    "action_space": envs.action_space,
                },
                print_system_info=True,
                load_torch_params_only=True,  # new argument to make sure config isn't overwritten
            )


    assert model is not None, "Model is None. Something went wrong."

    return model, model_class, policy_kwargs


def generate_callback_list(args: DictConfig, eval_callback: EvalCallback):
    if args.wandb:
        customwandbcallback = CustomWandbCallback()
        callback = CallbackList([eval_callback, customwandbcallback])
    else:
        callback = eval_callback
    return callback


if __name__ == "__main__":
    main()
