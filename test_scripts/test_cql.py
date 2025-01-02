import joblib
from gym import Env, spaces
from offline_rl_algorithms.offline_replay_buffers import H5ReplayBuffer
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
import torch as th
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch as th
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to get rid of the warning message
from stable_baselines3.common.vec_env import DummyVecEnv

from typing import Any, Dict

import torch as th

import os
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
from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from offline_rl_algorithms.wandb_logger import WandBLogger
from offline_rl_algorithms.callbacks import CustomWandbCallback, OfflineEvalCallback

from encoders.xclip_encoder import XCLIPEncoder


from envs.metaworld_envs.metaworld import (
    create_wrapped_env,
    instruction_to_environment,
    environment_to_instruction,
)


from stable_baselines3.common.policies import ActorCriticPolicy


def parse_entropy_term(value):
    try:
        return float(value)
    except ValueError:
        return value


def generate_callback_list(args, eval_callback: EvalCallback):
    if args.wandb:
        customwandbcallback = CustomWandbCallback()
        callback = CallbackList([eval_callback, customwandbcallback])
    else:
        callback = eval_callback
    return callback


def str2bool(v):
    # because argparse is trash
    # used for parsing boolean arguments
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--algo",
        type=str,
        default="iql",
        choices=["ppo", "sac", "cql", "calibrated_cql", "iql", "bc"],
    )
    parser.add_argument("--text_string", type=str, default="opening window")
    parser.add_argument("--dir_add", type=str, default="")
    # parser.add_argument("--env_id", type=str, default="window-open-v2")
    parser.add_argument("--offline_training_steps", type=int, default=100000)
    parser.add_argument("--total_time_steps", type=int, default=1000000)
    parser.add_argument("--n_envs", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_note", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_freq", default=50000, type=int, help="online eval frequency"
    )
    parser.add_argument(
        "--video_freq", default=50000, type=int, help="online video frequency"
    )
    parser.add_argument("--succ_end", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--pca_path", type=str, default=None)
    parser.add_argument("--transform_base_path", type=str, default=None)
    parser.add_argument("--transform_model_path", type=str, default=None)
    parser.add_argument("--random_reset", action="store_true")
    parser.add_argument("--time", action="store_false")
    parser.add_argument("--ignore_language", action="store_true")
    parser.add_argument(
        "--mix_buffers",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
    )
    parser.add_argument("--offline_h5_path", type=str, default=None)

    parser.add_argument(
        "--train_orcale", action="store_true"
    )  # load latent from h5 file
    parser.add_argument("--warm_up_runs", type=int, default=0)
    parser.add_argument("--norm_input", action="store_true")
    parser.add_argument("--norm_output", action="store_true")
    parser.add_argument("--time_reward", type=float, default=1.0)
    parser.add_argument("--threshold_reward", action="store_true")
    parser.add_argument("--entropy_term", type=parse_entropy_term, default="auto")
    parser.add_argument("--time_penalty", type=float, default=0.0)
    parser.add_argument("--succ_bonus", type=float, default=0.0)
    parser.add_argument(
        "--xclip_model", type=str, default="microsoft/xclip-base-patch16-zero-shot"
    )
    parser.add_argument("--frame_length", type=int, default=32)
    parser.add_argument("--exp_name_end", type=str, default="triplet_hard_neg")
    parser.add_argument(
        "--sparse_only",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
    )
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--obs_env", action="store_true")

    args = parser.parse_args()
    return args


# Return training and evaluation envs
def create_envs(args, env_id, text_instruction, use_simulator_reward):
    # compute a language feature
    encoder = XCLIPEncoder()
    lang_feat = encoder.encode_text(text_instruction)
    lang_feat = lang_feat.to("cpu").detach().squeeze()

    ignore_language = args.ignore_language

    if ignore_language:
        lang_feat = None

    if args.n_envs > 1:
        envs = SubprocVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=use_simulator_reward,
                    goal_observable=True,
                )
                for i in range(args.n_envs)
            ]
        )
    else:
        envs = DummyVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=use_simulator_reward,
                    goal_observable=True,
                )
            ]
        )

    if args.n_envs > 1:
        eval_env = SubprocVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=True,
                    monitor=True,
                    goal_observable=True,
                )
                for i in range(args.n_envs)
            ]
        )  # KitchenEnvDenseOriginalReward(time=True)
    else:
        eval_env = DummyVecEnv(
            [
                create_wrapped_env(
                    env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=True,
                    monitor=True,
                    goal_observable=True,
                )
            ]
        )  # KitchenEnvDenseOriginalReward(time=True)

    return envs, eval_env


def get_policy_algorithm(args, envs, log_dir):
    # We don't need as large of a network there is no language
    if args.ignore_language:
        policy_kwargs = {
            "net_arch": [256, 256],
        }
    else:
        policy_kwargs = {
            "net_arch": dict(pi=[256, 256], qf=[256, 256]),
            # "net_arch": dict(pi=[128, 128], qf=[256, 128]),
            "policy_layer_norm": True,
            "critic_layer_norm": True,
            # 'activation_fn': nn.Sequential(nn.ReLU(), nn.LayerNorm(256))
        }

    if args.algo.lower() == "ppo":
        model_class = PPO
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                n_steps=args.n_steps,
                batch_size=args.n_steps * args.n_envs,
                n_epochs=1,
                ent_coef=args.entropy_term,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif args.algo.lower() == "sac":
        model_class = SAC
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                # batch_size=args.n_steps * args.n_envs,
                ent_coef="auto",
                buffer_size=args.total_time_steps,
                learning_starts=4000,
                seed=args.seed,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif args.algo.lower() in ["cql", "calibrated_ql"]:
        use_calibrated_cql = args.algo.lower() == "calibrated_ql"
        model_class = CQL
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                ent_coef="auto",
                buffer_size=args.total_time_steps,
                learning_starts=0,
                seed=args.seed,
                min_q_weight=5.0,
                min_q_temp=1.0,
                use_calibrated_q=use_calibrated_cql,
                # learning_rate=0.0001,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    elif args.algo.lower() == "iql":
        model_class = IQL
        # import stable_baselines3

        # action_noise = stable_baselines3.common.noise.OrnsteinUhlenbeckActionNoise(
        #    mean=np.ones(4) * 5, sigma=1
        # )
        # n_actions = envs.action_space.shape[-1]
        # action_noise = stable_baselines3.common.noise.NormalActionNoise(
        #    mean=np.zeros(n_actions), sigma=0.1 * n_actions
        # )
        action_noise = None
        # policy = SACPolicy(observation_space=envs.observation_space, action_space=envs.action_space, net_arch=[32, 32], lr_schedule=None)
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=args.total_time_steps,
                learning_starts=0,
                seed=args.seed,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif args.algo.lower() == "bc":
        model_class = BC
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=args.total_time_steps,
                learning_starts=0,
                seed=args.seed,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    return model, model_class


def main():
    global args
    global log_dir
    args = get_args()

    # set seed
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"

    experiment_name = f"test_offline_rl_{args.algo}"
    # if args.pca_path != None:
    #     experiment_name = "ep500_PCA_" + "xclip_textTRANS_" + args.algo + "_" + args.env_id
    # else:
    #     experiment_name = "ep500_NOPCA_" +"xclip_textTRANS_" + args.algo + "_" + args.env_id

    # experiment_name = args.algo + "_" + args.env_id
    if args.train_orcale:
        experiment_name = experiment_name + "_Oracle"
    if args.threshold_reward:
        experiment_name = experiment_name + "_Thld"
    if args.succ_end:
        experiment_name = experiment_name + "_SuccEnd"
    experiment_name = experiment_name + args.exp_name_end
    run_group = experiment_name + "NEW"
    wandb.disabled = True

    if args.wandb:
        run = wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group=run_group,
            config=args,
            name=experiment_name,
            monitor_gym=True,
            sync_tensorboard=True,
            notes=args.wandb_note,
        )

        # column1 = ["text_string"]
        # table1 = wandb.Table(columns=column1)
        # table1.add_data([args.text_string])

        # column2 = ["env_id"]
        # table2 = wandb.Table(columns=column2)
        # table2.add_data([args.env_id])
        # self.logger.record({"text_string": table1, "env_id": table2})

    log_dir = f"logs/baseline_logs/{experiment_name}"

    # args.log_dir = log_dir # temporary hopefully

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set training and test tasks, by language instruction (for now)
    # offline_tasks = ["assembling", "picking bin", "closing box", "pressing button", "opening window"]
    offline_tasks = ["opening window"]

    online_task_env_id = instruction_to_environment[args.text_string]
    online_task_string = args.text_string

    envs, eval_env = create_envs(
        args,
        online_task_env_id,
        online_task_string,
        use_simulator_reward=(not args.sparse_only),
    )

    model, model_class = get_policy_algorithm(args, envs, log_dir)

    # Set eval freq and video freq if not set
    eval_freq = args.offline_training_steps * args.n_envs // (10)
    video_freq = args.offline_training_steps * args.n_envs // 10
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

    callback_list = generate_callback_list(args, eval_callback)

    # Create the logger
    wandb_logger = WandBLogger()
    model.set_logger(wandb_logger)

    ### Learn offline ###
    # if isinstance(model, OfflineRLAlgorithm):
    if False:
        # h5_path = "updated_trajs.h5"
        # h5_path = 'data/h5_buffers/updated_trajs/metaworld_dataset_sparse_only.h5'
        # h5_path = 'data/h5_buffers/updated_trajs/metaworld_window_traj_sparse_only.h5'
        # h5_path = 'data/h5_buffers/updated_trajs/metaworld_window_traj_orig_reward.h5'
        if args.offline_h5_path is None:
            default_h5_path = (
                "data/h5_buffers/updated_trajs/metaworld_traj_100_demos_orig_reward.h5"
            )
            print("There is no h5 path provided. Defaulting to", default_h5_path)
            h5_path = default_h5_path
        else:
            h5_path = args.offline_h5_path

        use_language = not args.ignore_language

        buffer = H5ReplayBuffer(
            h5_path,
            use_language_embeddings=use_language,
            success_bonus=args.succ_bonus,
            sparsify_rewards=args.sparse_only,
            filter_instructions=offline_tasks,
        )
        model.learn_offline(
            offline_replay_buffer=buffer,
            train_steps=args.offline_training_steps,
            callback=callback_list,
            batch_size=256,
        )

    logger = model.logger

    online_eval_freq = args.eval_freq // args.n_envs  # // args.nenvsto
    online_video_freq = args.video_freq // args.n_envs
    eval_callback.eval_freq = online_eval_freq
    eval_callback.video_freq = online_video_freq

    model.learn(
        total_timesteps=int(args.total_time_steps),
        callback=callback_list,
        logger=logger,
        progress_bar=True,
    )
    model.save(f"{log_dir}/{experiment_name}")

    # Evaluate the agent
    # load the best model
    model = model_class.load(f"{log_dir}/best_model")
    # success_rate = eval_policys(args, MetaworldDense, model)

    # if args.wandb:
    #     self.logger.record({"eval_SR/evaluate_succ": success_rate}, step = 0)
    if args.wandb:
        run.finish()


if __name__ == "__main__":
    main()
