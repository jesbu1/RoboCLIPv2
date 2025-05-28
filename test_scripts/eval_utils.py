from models.reward_model.base_reward_model import BaseRewardModel
from new_task_annotation_v2 import eval_gt_annotation, train_gt_annotation
from envs.metaworld_envs.metaworld import create_wrapped_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import torch as th
import wandb
import numpy as np
import imageio
import _frozen_importlib
import io


def offline_eval(policy, reward_model: BaseRewardModel, image_encoder, rollout_num = 25):
    # Extract configuration
    # env_id = env_config.env_id
    # env_id = instruction_to_environment[env_config.text_string]
    # text_instruction = env_config.text_string

    eval_envs = list(eval_gt_annotation.keys())
    train_envs = list(train_gt_annotation.keys())
    # eval_envs = eval_envs + train_envs

    wandb_log = {}

    for env_id in tqdm(eval_envs):
        if env_id in eval_gt_annotation:
            text_instruction = eval_gt_annotation[env_id]
        else:
            text_instruction = train_gt_annotation[env_id]
        with th.no_grad():
            lang_feat_policy = reward_model.encode_text_for_policy(text_instruction).squeeze()
            # lang_feat = th.from_numpy(lang_feat).squeeze()
            if reward_model.name != "GVLRewardModel" and reward_model.name != "VLCRewardModel":
                lang_feat_reward = reward_model.encode_text(text_instruction).squeeze()
                print("Lang feat reward shape", lang_feat_reward.shape)
            else:
                lang_feat_reward = text_instruction

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
            render = False
            for rollout_id in range(rollout_num):
                obs = eval_env.reset()
                if rollout_id == rollout_num - 1:
                    render = True
                if render:
                    img_list = []
                    img = eval_env.render(mode="rgb_array")
                    img_list.append(img)
                for _ in range(
                    eval_env.get_attr("max_episode_steps")[0]
                ):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    if render:
                        img = eval_env.render(mode="rgb_array")
                        img_list.append(img)
                    # import pdb; pdb.set_trace()
                    if info[0]["is_success"]:
                        success_num += 1
                        break
                
                if render:

                    video_buffer = io.BytesIO()
                    # save as gif

                    with imageio.get_writer(video_buffer, format='mp4', fps=20) as writer:
                        for frame in img_list:
                            writer.append_data(frame)
                    video_buffer.seek(0)
                    wandb.log({f"evaluation_video/eval_video/{env_id}": wandb.Video(video_buffer, fps=20, format="mp4")})


                wandb_log[f"offline_eval/eval_task/{env_id}_success_rate"] = success_num / rollout_num

    # for env_id in tqdm(train_envs):
    #     if env_id in eval_gt_annotation:
    #         text_instruction = eval_gt_annotation[env_id]
    #     else:
    #         text_instruction = train_gt_annotation[env_id]
    #     with th.no_grad():
    #         lang_feat_policy = reward_model.encode_text_for_policy(text_instruction).squeeze()
    #         lang_feat_reward = reward_model.encode_text(text_instruction).squeeze()

    #         eval_env = DummyVecEnv(
    #                     [
    #                         create_wrapped_env(
    #                             env_id,
    #                             reward_model=reward_model,
    #                             image_encoder=image_encoder,
    #                             language_features_policy=lang_feat_policy,
    #                             language_features_reward=lang_feat_reward,
    #                             monitor=True,
    #                             goal_observable=True,
    #                             is_state_based=False,
    #                             mode="eval",
    #                             use_proprio=True,
    #                         )
    #                     ]
    #                 )

    #         success_num = 0
    #         for rollout_id in range(rollout_num):
    #             obs = eval_env.reset()
    #             for _ in range(
    #                 eval_env.get_attr("max_episode_steps")[0]
    #             ):
    #                 action, _ = policy.predict(obs, deterministic=True)
    #                 obs, reward, done, info = eval_env.step(action)
    #                 # import pdb; pdb.set_trace()
    #                 if info[0]["is_success"]:
    #                     success_num += 1
    #                     break
    #         wandb_log[f"offline_eval/train_task/{env_id}_success_rate"] = success_num / rollout_num

    wandb.log(wandb_log)
    
                





            
            









