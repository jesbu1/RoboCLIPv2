from models.reward_model.base_reward_model import BaseRewardModel
from new_task_annotation_v2 import eval_gt_annotation
from envs.metaworld_envs.metaworld import create_wrapped_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import torch as th
import wandb


def offline_eval(policy, reward_model: BaseRewardModel, image_encoder, rollout_num = 10):
    # Extract configuration
    # env_id = env_config.env_id
    # env_id = instruction_to_environment[env_config.text_string]
    # text_instruction = env_config.text_string

    eval_envs = list(eval_gt_annotation.keys())
    
    wandb_log = {}

    for env_id in tqdm(eval_envs):
        text_instruction = eval_gt_annotation[env_id]
        with th.no_grad():
            lang_feat_policy = reward_model.encode_text_for_policy(text_instruction).squeeze()
            lang_feat_reward = reward_model.encode_text(text_instruction).squeeze()

            eval_env = DummyVecEnv(
                        [
                            create_wrapped_env(
                                env_id,
                                reward_model=reward_model,
                                image_encoder=image_encoder,
                                language_features_policy=lang_feat_policy,
                                language_features_reward=lang_feat_reward,
                                monitor=True,
                                goal_observable=True,
                                is_state_based=False,
                                mode="eval",
                                use_proprio=True,
                            )
                        ]
                    )

            success_num = 0
            for rollout_id in range(rollout_num):
                obs = eval_env.reset()
                for _ in range(
                    eval_env.get_attr("max_episode_steps")[0]
                ):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    # import pdb; pdb.set_trace()
                    if info[0]["is_success"]:
                        success_num += 1
                        break
            wandb_log[f"offline_eval/{env_id}_success_rate"] = success_num / rollout_num
    wandb.log(wandb_log)
    
                





            
            









