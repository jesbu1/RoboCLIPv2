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

from encoders.xclip_encoder import XCLIPEncoder


from envs.metaworld_envs.metaworld import create_wrapped_env


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


class OfflineEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(OfflineEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq
        # we need to overide num_timesteps as EvalCallback uses it to align the built in logger's x-axis
        # we are using wandb so now we're using self.n_calls as the step for everything
        self.num_timesteps = lambda x: self.n_calls  # convert num_timst

    def _on_step(self) -> bool:
        # print(self.n_calls, self.n_calls % self.video_freq)
        # Log policy gradients
        policy_gradients = [
            param.grad.view(-1).detach().cpu().numpy()  # Flatten each gradient tensor
            for param in self.model.policy.actor.parameters()
            if param.grad is not None
        ]
        if len(policy_gradients) != 0:

            all_gradients = np.concatenate(policy_gradients)
            self.logger.record("grad/policy_histogram", wandb.Histogram(all_gradients))
        if hasattr(self.model, "v_net"):
            # Log critic gradients
            critic_gradients = [
                param.grad.view(-1)
                .detach()
                .cpu()
                .numpy()  # Flatten each gradient tensor
                for param in self.model.policy.critic.parameters()
                if param.grad is not None
            ]
            if len(critic_gradients) != 0:
                all_gradients = np.concatenate(critic_gradients)
                self.logger.record(
                    "grad/critic_histogram", wandb.Histogram(all_gradients)
                )

            # Log critic_target gradients
            critic_target_gradients = [
                param.grad.view(-1)
                .detach()
                .cpu()
                .numpy()  # Flatten each gradient tensor
                for param in self.model.policy.critic_target.parameters()
                if param.grad is not None
            ]
            if len(critic_target_gradients) != 0:
                all_gradients = np.concatenate(critic_target_gradients)
                self.logger.record(
                    "grad/critic_target_histogram", wandb.Histogram(all_gradients)
                )

            # Log v_net gradients
            v_net_gradients = [
                param.grad.view(-1)
                .detach()
                .cpu()
                .numpy()  # Flatten each gradient tensor
                for param in self.model.v_net.parameters()
                if param.grad is not None
            ]
            if len(v_net_gradients) != 0:
                all_gradients = np.concatenate(v_net_gradients)
                self.logger.record(
                    "grad/v_net_histogram", wandb.Histogram(all_gradients)
                )
            # Log critic weights
            critic_weights = [
                param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
                for param in self.model.policy.critic.parameters()
            ]
            if len(critic_weights) != 0:
                all_weights = np.concatenate(critic_weights)
                self.logger.record(
                    "weights/critic_histogram", wandb.Histogram(all_weights)
                )

        # Log critic_target weights
        critic_target_weights = [
            param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
            for param in self.model.policy.critic_target.parameters()
        ]
        if len(critic_target_weights) != 0:
            all_weights = np.concatenate(critic_target_weights)
            self.logger.record(
                "weights/critic_target_histogram", wandb.Histogram(all_weights)
            )
            # Log critic_target weights
            critic_target_weights = [
                param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
                for param in self.model.policy.critic_target.parameters()
            ]
            if len(critic_target_weights) != 0:
                all_weights = np.concatenate(critic_target_weights)
                self.logger.record(
                    "weights/critic_target_histogram", wandb.Histogram(all_weights)
                )

            # Log v_net weights
            v_net_weights = [
                param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
                for param in self.model.v_net.parameters()
            ]
            if len(v_net_weights) != 0:
                all_weights = np.concatenate(v_net_weights)
                self.logger.record(
                    "weights/v_net_histogram", wandb.Histogram(all_weights)
                )

        # Log policy weights
        actor_weights = [
            param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
            for param in self.model.policy.actor.parameters()
        ]
        if len(actor_weights) != 0:
            all_weights = np.concatenate(actor_weights)
            self.logger.record("weights/policy_histogram", wandb.Histogram(all_weights))

        # breakpoint()
        if (
            self.video_freq > 0 and self.n_calls % self.video_freq == 0
        ) or self.n_calls == 1:
            video_buffer = self.record_video()
            # self.logger.record({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            self.logger.record(
                "eval/evaluation_video", wandb.Video(video_buffer, fps=20, format="mp4")
            )
            # self.logger.record({f"eval/evaluate_succ": success}, step = self.n_calls)
            print("video logged")

        self.logger.record("num_timesteps", self.num_timesteps)

        result = super(OfflineEvalCallback, self)._on_step()

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        # success = 0
        # breakpoint()

        # print the first layer's weight of self.model.policy
        print(self.model.policy.actor.latent_pi[0].weight[0][:10])

        for _ in range(128):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode="rgb_array")
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=False)
            # action += np.random.normal(5, 0.1, size=action.shape)
            # action[0] += 5
            # print(action)
            obs, _, _, info = self.eval_env.step(action)
            # print(type(info))
            # print(info)
            # if info['success']:
            #     success = 1
            #     break

        video_buffer = io.BytesIO()

        with imageio.get_writer(video_buffer, format="mp4", fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_buffer.seek(0)
        return video_buffer


class CustomWandbCallback(WandbCallback):
    def _on_step(self):
        if "metrics" in self.locals:
            self.logger.record_dict(self.locals["metrics"])
        self.logger.dump(
            self.n_calls
        )  # this ensures that dump gets called, otherwise it's only called in EvalCallback whenever an eval happens


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
    parser.add_argument("--env_id", type=str, default="window-open-v2")
    parser.add_argument("--offline_training_steps", type=int, default=100000)
    parser.add_argument("--total_time_steps", type=int, default=1000000)
    parser.add_argument("--n_envs", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
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
    parser.add_argument("--mix_buffers", action="store_true")
    parser.add_argument("--offline_h5_path", type=str, default=None)

    parser.add_argument(
        "--train_orcale", action="store_true"
    )  # load latent from h5 file
    parser.add_argument("--warm_up_runs", type=int, default=0)
    parser.add_argument("--project_reward", action="store_true")
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
    parser.add_argument("--sparse_only", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--obs_env", action="store_true")

    args = parser.parse_args()
    return args


class SingleLayerMLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


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
    if args.project_reward:
        experiment_name = experiment_name + "_ProjReward"
    # if args.norm_input:
    #     experiment_name = experiment_name + "_NormIn"
    # if args.norm_output:
    #     experiment_name = experiment_name + "_NormOut"
    # if args.time_reward != 1.0:
    #     experiment_name = experiment_name + "_XReward" + str(args.time_reward)
    # if args.time:
    #     experiment_name = experiment_name + "_Time"
    # else:
    #     experiment_name = experiment_name + "_NoTime"
    if args.succ_end:
        experiment_name = experiment_name + "_SuccEnd"
    # if args.random_reset:
    #     experiment_name = experiment_name + "_RandReset"

    # if args.succ_bonus > 0:
    #     experiment_name = experiment_name + "_SuccBonus" + str(args.succ_bonus)
    # if args.time_penalty > 0:
    #     experiment_name = experiment_name + "_TimePenalty" + str(args.time_penalty)
    # if args.algo.lower() == 'sac':
    # experiment_name = experiment_name + "_Entropy" + str(args.entropy_term)
    experiment_name = experiment_name + args.exp_name_end
    run_group = experiment_name + "NEW"
    # experiment_name = experiment_name + "_" + str(args.seed) + "NEW"
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
        )

        # column1 = ["text_string"]
        # table1 = wandb.Table(columns=column1)
        # table1.add_data([args.text_string])

        # column2 = ["env_id"]
        # table2 = wandb.Table(columns=column2)
        # table2.add_data([args.env_id])
        # self.logger.record({"text_string": table1, "env_id": table2})

    # log_dir = f"/scr/jzhang96/logs/baseline_logs/{experiment_name}"
    log_dir = f"logs/baseline_logs/{experiment_name}"
    # log_dir = f"/home/jzhang96/logs/baseline_logs/{experiment_name}"

    args.log_dir = log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # compute a language feature
    encoder = XCLIPEncoder()
    lang_feat = encoder.encode_text(args.text_string)
    lang_feat = lang_feat.to("cpu").detach().squeeze()

    ignore_language = args.ignore_language
    use_language = not ignore_language

    if ignore_language:
        lang_feat = None

    if args.n_envs > 1:
        envs = SubprocVecEnv(
            [
                create_wrapped_env(
                    args.env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=True,
                    goal_observable=True,
                )
                for i in range(args.n_envs)
            ]
        )
    else:
        envs = DummyVecEnv(
            [
                create_wrapped_env(
                    args.env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=False,
                )
            ]
        )

    # We don't need as large of a network there is no language
    if ignore_language:
        policy_kwargs = {
            "net_arch": [256, 256],
        }
    else:
        policy_kwargs = {
            "net_arch": dict(pi=[512, 256], qf=[512, 256, 256]),
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
                learning_starts=4000,
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
                learning_starts=4000,
                seed=args.seed,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                mix_offline_online_buffers=args.mix_buffers,
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
                learning_starts=4000,
                seed=args.seed,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    if args.n_envs > 1:
        eval_env = SubprocVecEnv(
            [
                create_wrapped_env(
                    args.env_id,
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
                    args.env_id,
                    language_features=lang_feat,
                    success_bonus=args.succ_bonus,
                    use_simulator_reward=True,
                    monitor=True,
                    goal_observable=True,
                )
            ]
        )  # KitchenEnvDenseOriginalReward(time=True)

    # Set eval freq and video freq if not set
    # eval will be done 10 times
    eval_freq = args.offline_training_steps * args.n_envs // 80
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

    # load the offline replay buffer
    if isinstance(model, OfflineRLAlgorithm):
        # if False:
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
        buffer = H5ReplayBuffer(
            h5_path,
            use_language_embeddings=use_language,
            success_bonus=args.succ_bonus,
            sparsify_rewards=True,
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
    )
    model.save(f"{log_dir}/{experiment_name}")

    # Evaluate the agent
    # load the best model
    model = model_class.load(f"{log_dir}/best_model")
    # success_rate = eval_policys(args, MetaworldDense, model)

    # if args.wandb:
    #     self.logger.record({"eval_SR/evaluate_succ": success_rate}, step = 0)


if __name__ == "__main__":
    main()
