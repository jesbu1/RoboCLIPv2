#!/usr/bin/env python3
"""
Evaluation script for RoboCLIPv2 models.
This script loads a trained model and evaluates it for N episodes.
"""

import os
import sys
import time
import gym
import torch as th
import numpy as np
import wandb
import json
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

# Import the necessary modules from the project
from offline_rl_algorithms.iql import IQL
from offline_rl_algorithms.bc import BC
from offline_rl_algorithms.rlpd import RLPD
from offline_rl_algorithms.cql import CQL
from offline_rl_algorithms.wandb_logger import WandBLogger

# Import functions from test_iql.py
from test_scripts.test_iql import parse_reward_model, create_envs, create_exp_name

import io
import imageio


def load_model(cfg, model_path, env, model_type="rlpd", offline_algo=None):
    """
    Load a model from the specified path.

    Args:
        model_path: Path to the model directory
        env: Environment to use with the model
        model_type: Type of model to load (rlpd, iql, bc, cql)
        offline_algo: Offline algorithm for RLPD

    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}")

    # Check if this is a custom-saved model (look for component files)
    policy_path = os.path.join(model_path, "policy.pth")
    actor_path = os.path.join(model_path, "actor.pth")
    critic_path = os.path.join(model_path, "critic.pth")
    v_net_path = os.path.join(model_path, "v_net.pth")
    params_path = os.path.join(model_path, "params.json")

    is_custom_saved = os.path.exists(params_path) and (
        os.path.exists(policy_path)
        or os.path.exists(actor_path)
        or os.path.exists(critic_path)
        or os.path.exists(v_net_path)
    )

    if is_custom_saved:
        print("Detected custom-saved model, loading components individually")

        # Load parameters from JSON
        with open(params_path, "r") as f:
            params = json.load(f)

        # We need to use the same policy type and network architecture as the saved model
        # Get these from the config
        policy_type = "RnnMlpPolicy"  # Default to RnnMlpPolicy if not specified

        # Create policy_kwargs with the same architecture as the saved model
        policy_kwargs = {
            "net_arch": dict(
                pi=[1024, 768, 512],  # Default architecture from the config
                qf=[1024, 1024, 512],
            ),
            "policy_layer_norm": True,
            "critic_layer_norm": True,
        }

        if "action_chunk_size" in params:
            policy_kwargs["action_sequence_length"] = params["action_chunk_size"]
        else:
            # Default action chunk size from the config
            policy_kwargs["action_sequence_length"] = 30

        # Create a new model instance based on the algorithm type
        if model_type.lower() == "rlpd":
            if offline_algo is None:
                # For RLPD, we need to create an offline algorithm first
                if "iql" in model_path.lower():
                    offline_algo = IQL(
                        policy_type, env, verbose=1, policy_kwargs=policy_kwargs
                    )
                elif "bc" in model_path.lower():
                    offline_algo = BC(
                        policy_type, env, verbose=1, policy_kwargs=policy_kwargs
                    )
                elif "cql" in model_path.lower():
                    offline_algo = CQL(
                        policy_type, env, verbose=1, policy_kwargs=policy_kwargs
                    )

            model = RLPD(
                policy_type,
                env,
                offline_algo=offline_algo,
                verbose=1,
                policy_kwargs=policy_kwargs,
            )
        elif model_type.lower() == "iql":
            model = IQL(policy_type, env, verbose=1, policy_kwargs=policy_kwargs)
        elif model_type.lower() == "bc":
            model = BC(policy_type, env, verbose=1, policy_kwargs=policy_kwargs)
        elif model_type.lower() == "cql":
            model = CQL(policy_type, env, verbose=1, policy_kwargs=policy_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load policy weights if available
        if os.path.exists(policy_path):
            print(f"Loading policy weights from {policy_path}")
            policy_state_dict = th.load(policy_path)

            # Check if we need to do a strict load or a non-strict load
            try:
                model.policy.load_state_dict(policy_state_dict)
            except RuntimeError as e:
                print(f"Warning: Could not load policy weights strictly. Error: {e}")
                print("Attempting to load with strict=False...")
                model.policy.load_state_dict(policy_state_dict, strict=False)

        # Load actor weights if available
        if os.path.exists(actor_path) and hasattr(model, "actor"):
            print(f"Loading actor weights from {actor_path}")
            actor_state_dict = th.load(actor_path)

            try:
                model.actor.load_state_dict(actor_state_dict)
            except RuntimeError as e:
                print(f"Warning: Could not load actor weights strictly. Error: {e}")
                print("Attempting to load with strict=False...")
                model.actor.load_state_dict(actor_state_dict, strict=False)

        # Load critic weights if available
        if os.path.exists(critic_path) and hasattr(model, "critic"):
            print(f"Loading critic weights from {critic_path}")
            critic_state_dict = th.load(critic_path)

            try:
                model.critic.load_state_dict(critic_state_dict)
            except RuntimeError as e:
                print(f"Warning: Could not load critic weights strictly. Error: {e}")
                print("Attempting to load with strict=False...")
                model.critic.load_state_dict(critic_state_dict, strict=False)

        # Load value network weights if available (for IQL)
        if os.path.exists(v_net_path) and hasattr(model, "v_net"):
            print(f"Loading value network weights from {v_net_path}")
            v_net_state_dict = th.load(v_net_path)

            try:
                model.v_net.load_state_dict(v_net_state_dict)
            except RuntimeError as e:
                print(
                    f"Warning: Could not load value network weights strictly. Error: {e}"
                )
                print("Attempting to load with strict=False...")
                model.v_net.load_state_dict(v_net_state_dict, strict=False)

        return model
    else:
        # Use standard loading method for models saved with the built-in save method
        print("Using standard model loading method")

        if model_type.lower() == "rlpd":
            if offline_algo is None:
                # Try to load the offline algorithm first
                offline_path = f"{model_path}_rlpd_offline"
                if os.path.exists(offline_path):
                    print(f"Loading offline algorithm from {offline_path}")
                    # offline_algo = IQL.load(offline_path, env=env)
                else:
                    print(f"Warning: Offline algorithm not found at {offline_path}")

            model = RLPD(
                cfg.model.policy_type,
                env,
                offline_algo=None,
                verbose=1,
                # tensorboard_log=log_dir,
                # buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                # seed=args.seed,
                # action_noise=action_noise,  # should be null
                ent_coef=cfg.general_training.entropy_term,
                # policy_kwargs=policy_kwargs,
                learning_rate=cfg.general_training.learning_rate,
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
                action_chunk_size=cfg.general_training.action_chunk_size,
                success_bonus=cfg.reward_model.success_bonus,
                gradient_steps=cfg.online_training.gradient_steps,
            )

            model = model.load(
                path=model_path,
                env=env,
                offline_algo=offline_algo,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )
        elif model_type.lower() == "iql":
            model = IQL.load(
                path=model_path,
                env=env,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )
        elif model_type.lower() == "bc":
            model = BC(policy_type, env, verbose=1, policy_kwargs=policy_kwargs)
            model.load(
                path=model_path,
                env=env,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )
        elif model_type.lower() == "cql":
            model = CQL.load(path=model_path, env=env)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model


def evaluate_model(
    model, env, n_episodes=10, deterministic=True, render=False, use_wandb=False
):
    """
    Evaluate a model for n_episodes.

    Args:
        model: The model to evaluate
        env: Environment to evaluate in
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        use_wandb: Whether to log to wandb

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    # Set up video recording if wandb is enabled
    video_frames = []

    for i in tqdm(range(n_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        first_step = True
        step_rewards = []

        # For recording video
        episode_frames = []

        while not done:
            action, _ = model.predict(
                obs, deterministic=True, episode_start=np.array([first_step])
            )
            obs, reward, done, info = env.step(action)

            # Capture frame for video if using wandb
            if use_wandb or render:
                # Get the image from observation if available
                if (
                    isinstance(obs, dict)
                    and "observation" in obs
                    and "images" in obs["observation"]
                ):
                    frame = obs["observation"]["images"]["main"]
                    if frame is not None:
                        # Convert to RGB format if needed
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            episode_frames.append(frame)
                elif hasattr(env, "render"):
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        episode_frames.append(frame)

            episode_reward += reward
            step_rewards.append(reward)
            episode_length += 1
            first_step = False

            # Log per-step metrics if using wandb
            if use_wandb:
                wandb.log(
                    {
                        "step": episode_length,
                        "reward": reward,
                    }
                )

            if render:
                env.render()

            if done:
                success = info[0].get("success", False)
                episode_successes.append(float(success))
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Log per-episode metrics if using wandb
                if use_wandb:
                    wandb.log(
                        {
                            "episode": i,
                            "episode_reward": episode_reward,
                            "episode_length": episode_length,
                            "episode_success": float(success),
                            "step_rewards_hist": wandb.Histogram(
                                np.array(step_rewards)
                            ),
                        }
                    )

                    # Save video of the episode
                    if episode_frames:
                        # video_frames.append(np.array(episode_frames))

                        video_buffer = io.BytesIO()
                        with imageio.get_writer(
                            video_buffer, format="mp4", fps=20
                        ) as writer:
                            for frame in episode_frames:
                                writer.append_data(frame)

                        video_buffer.seek(0)

                        # Log the video of this episode
                        wandb.log(
                            {
                                "episode_video": wandb.Video(
                                    video_buffer,
                                    fps=20,
                                    format="mp4",
                                    # caption=f"Episode {i}: Reward={np.sum(episode_reward):.2f}, Success={success}",
                                )
                            }
                        )

                break

    # Calculate metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_success = np.mean(episode_successes)
    mean_length = np.mean(episode_lengths)

    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": mean_success,
        "mean_episode_length": mean_length,
        "num_episodes": n_episodes,
    }

    # Log final metrics if using wandb
    if use_wandb:
        wandb.log(metrics)

    return metrics


@hydra.main(config_path="../configs", config_name="base_config")
def main(cfg: DictConfig):
    """
    Main function for evaluating a trained model.

    Args:
        cfg: Hydra configuration
    """
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    np.random.seed(cfg.evaluation.seed)
    th.manual_seed(cfg.evaluation.seed)

    # Extract configurations
    env_config = cfg.environment
    model_config = cfg.model
    logging_config = cfg.logging
    eval_config = cfg.evaluation

    # Setup wandb if enabled
    if eval_config.wandb:
        config_for_wandb = OmegaConf.to_container(cfg, resolve=True)
        experiment_name = f"eval_{create_exp_name(cfg)}"

        wandb.init(
            entity=logging_config.wandb_entity_name,
            project=logging_config.wandb_project_name,
            group=logging_config.wandb_group_name,
            name=experiment_name,
            config=config_for_wandb,
            monitor_gym=True,
            sync_tensorboard=True,
        )

        wandb_logger = WandBLogger()
    else:
        wandb_logger = None

    # Parse reward model
    reward_model = parse_reward_model(cfg.reward_model)

    # Create environments
    envs, eval_env = create_envs(cfg, reward_model, logger=wandb_logger)

    # Convert model path to absolute path if needed
    model_path = (
        to_absolute_path(eval_config.model_path)
        if not os.path.isabs(eval_config.model_path)
        else eval_config.model_path
    )

    # Load the model
    model = load_model(cfg, model_path, envs, model_type=eval_config.model_type)

    # Set the logger if available
    if wandb_logger is not None:
        model.set_logger(wandb_logger)

    print(f"Evaluating model for {eval_config.n_episodes} episodes...")

    # Evaluate the model
    metrics = evaluate_model(
        model=model,
        env=eval_env,
        n_episodes=eval_config.n_episodes,
        deterministic=eval_config.deterministic,
        render=eval_config.render,
        use_wandb=eval_config.wandb,
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Number of episodes: {metrics['num_episodes']}")
    print(f"Success rate: {metrics['success_rate']:.4f}")
    print(f"Mean reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"Mean episode length: {metrics['mean_episode_length']:.2f}")

    # Log to wandb if enabled
    if eval_config.wandb:
        wandb.log(metrics)
        wandb.finish()

    return metrics


if __name__ == "__main__":
    main()
