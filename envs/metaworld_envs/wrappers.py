import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gym import spaces
from typing import List
from memory_profiler import profile
from models.reward_model.base_reward_model import BaseRewardModel
from models.encoders.base_encoder import BaseEncoder
import wandb

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


def adjust_frames_xclip(frames, target_frame_count=32, processor=None):
    """
    Ensures same numbers of frames(32). returns a numpy array of shape (target_frame_count, 224, 224, 3)
    """
    frames = np.array(frames)
    frame_count = frames.shape[0]
    # print(f"frames number{frame_count}")
    # frames = th.from_numpy(frames)

    if len(frames) > target_frame_count:
        index = np.linspace(0, len(frames) - 1, target_frame_count, dtype=int)
        frames = frames[index]
    elif len(frames) < target_frame_count:
        last_frame = frames[-1]
        last_frame = np.expand_dims(last_frame, axis=0)
        for _ in range(target_frame_count - len(frames)):
            frames = np.concatenate([frames, last_frame])
    frames = frames[:, 240 - 112 : 240 + 112, 320 - 112 : 320 + 112, :]
    # frames = frames[None, :,:,:,:]
    frames = processor(videos=list(frames), return_tensors="pt")
    frames = frames["pixel_values"]
    return frames


def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()


# Wrapper for PCA
class PCAReducerWrapper(gym.Wrapper):
    def __init__(self, env, pca_model):
        super(PCAReducerWrapper, self).__init__(env)
        self.pca_model = pca_model
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.pca_model.n_components,),
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_pca = self.pca_model.transform(obs.reshape(1, -1)).flatten()
        return obs_pca, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.pca_model.transform(obs.reshape(1, -1)).flatten()


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, sparse=True, success_bonus=0.0):
        super(RewardWrapper, self).__init__(env)
        self.sparse = sparse
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.success_bonus = success_bonus

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Convert dense rewards to sparse
        sparse_reward = self.success_bonus if info.get("success", False) else 0.0
        if self.sparse:
            reward = sparse_reward
        else:
            reward = reward + sparse_reward

        return obs, reward, done, info


# Wrapper for Time-based Observations
class TimeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TimeWrapper, self).__init__(env)
        self.counter = 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_space.shape[0] + 1,),
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        t = self.counter / 128  # Assuming max steps is 500
        obs = np.concatenate([obs, [t]])
        self.counter += 1
        return obs, reward, done, info

    def reset(self):
        self.counter = 0
        obs = self.env.reset()
        return np.concatenate([obs, [0]])  # Add time as 0 at reset


# Wrapper for Language-based Observations
# All this environment does is change the observation space
# This will append a specific language feature to the observation
class LanguageWrapper(gym.Wrapper):
    def __init__(self, env, language_feature):
        super(LanguageWrapper, self).__init__(env)

        if isinstance(language_feature, th.Tensor):
            language_feature = language_feature.cpu().numpy()

        self.language_features = language_feature
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_space.shape[0] + len(self.language_features),),
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([obs, self.language_features])
        # print(f"obs after language wrapper: {obs.shape}")
        return obs, reward, done, info
    # @profile
    def reset(self):
        obs = self.env.reset()
        return np.concatenate([obs, self.language_features])


class LearnedRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_model: BaseRewardModel,
        encoder: BaseEncoder,
        language_features_reward: th.Tensor,
        is_state_based: bool = False,
        dense_eval: bool = False,
        use_proprio: bool = False,
    ):
        super(LearnedRewardWrapper, self).__init__(env)
        self.reward_model = reward_model
        self.image_encoder = encoder
        self.is_state_based = is_state_based
        self.use_proprio = use_proprio

        if self.is_state_based is False:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.image_encoder.img_output_dim + (4 if self.use_proprio else 0),
                ),
                dtype=np.float32,
            )

        self.past_observations = []
        self.raw_observations = []
        self.counter = 0

        self.dense_eval = dense_eval

        self.reward_at_every_step = self.reward_model.reward_at_every_step
        self.reward_divisor = self.reward_model.reward_divisor

        if language_features_reward is not None:
            self.reward_language_features = (
                th.Tensor(language_features_reward)
                .float()
                .to(self.reward_model.device)
                .unsqueeze(0)
            )
        else:
            print("Language features are not provided in the reward model")
            print(
                "This may be valid if the user is using sparse/dense reward in a single task"
            )
    # @profile #not here
    def step(self, action):
        self.counter += 1
        obs, original_reward, done, info = self.env.step(action)
        proprio = obs[0:4]

        encoded_image = None
        # IF the model is state-based or is dense/sparse reward, we can skip this
        if not (
            (self.is_state_based)
            and (
                self.reward_model.name == "sparse" or self.reward_model.name == "dense"
            )
        ):
            # if state-based, we can render every 10 steps
            if (self.is_state_based and self.counter % 10 == 0) or (
                not self.is_state_based
            ):
                image = self.env.render()
                # Input should be of shape (batch_size, num_frames, height, width, channels)
                # However, the input is of shape (height, width, channels)
                image_for_model = image[None, None, :, :, :]
                self.raw_observations.append(image_for_model)
                # encoded_image = self.reward_model.encode_images(
                #     image_for_model
                # ).squeeze()
                encoded_image = self.image_encoder.encode_images(
                    image_for_model
                ).squeeze()

        if self.is_state_based is False and encoded_image is not None:
            # obs = np.concatenate([obs, self.reward_model(obs)])
            obs = encoded_image

            if self.use_proprio:
                obs = np.concatenate([obs, proprio])

        if self.reward_model.name == "dense" or self.dense_eval:
            # reward = original_reward / self.reward_divisor
            reward = original_reward
            if info.get("success", False):
                reward += self.reward_model.success_bonus
                if self.dense_eval:
                    print(f"eval success reward: {reward}")
            # print(f"obs: {obs.shape}") # 772 = 768 + 4
            if self.dense_eval:
                wandb.log({
                    "eval/eval_original_reward": original_reward,
                    "eval/eval_reward_with_success_bonus": reward
                })
            return obs, reward, done, info
        # Check if this is sparse/dense reward
        elif self.reward_model.name == "sparse":
            sparse_reward = (
                self.reward_model.success_bonus if info.get("success", False) else 0.0
            )
            wandb.log({"train/sparse_reward": sparse_reward})
            # Note: No reward divisor for sparse reward.

            return obs, sparse_reward, done, info

        
        if encoded_image is not None:
            self.past_observations.append(encoded_image)

        assert (
            self.reward_language_features is not None
        ), "Language features are None in the reward model"
        if self.reward_at_every_step:
            # frames = [
            #             frame[ 
            #                 (frame.shape[0] - 224) // 2 : (frame.shape[0] + 224) // 2,
            #                 (frame.shape[1] - 224) // 2 : (frame.shape[1] + 224) // 2,
            #                 :3 
            #             ]
            #             for frame in self.raw_observations
            #         ]
            # print(f"frames shape: {frames.shape}")
            # frames_embeddings = self.reward_model.encode_images(
            #     th.tensor(frames).float().to(self.reward_model.device)
            # )
            # print(f"frames_embeddings shape: {frames_embeddings.shape}")
            stacked_sequence = np.stack(self.past_observations, axis=1)
            stacked_sequence = (
                th.from_numpy(stacked_sequence).float().to(self.reward_model.device)
            )

            reward = self.reward_model.calculate_rewards(
                self.reward_language_features, stacked_sequence
            )

        else:
            if done:
                # stacked_sequence = np.stack(self.past_observations, axis=0)
                # stacked_sequence = (
                #     th.from_numpy(stacked_sequence).float().to(self.reward_model.device)
                # )
                # print(f"stacked_sequence shape: {stacked_sequence.shape}")
                # print(f"raw_observations shape: {len(self.raw_observations)}")
                # print(f"raw_observations shape: {self.raw_observations[0].shape}")
                frames = [
                          frame[
                            :,
                            :, 
                            (frame.shape[2] - 224) // 2 : (frame.shape[2] + 224) // 2,
                            (frame.shape[3] - 224) // 2 : (frame.shape[3] + 224) // 2,
                            :3 
                        ]
                        for frame in self.raw_observations
                    ]
                frames = np.stack(frames, axis=1).squeeze(2)
                # print(f"frames shape: {frames.shape}") # (1, 128, 224, 224, 3)
                frames_embeddings = th.from_numpy(self.reward_model.encode_images(
                    frames
                )).unsqueeze(0)
                # print(f"frames_embeddings shape: {frames_embeddings.shape}") # (1, 32, 768)
                reward = self.reward_model.calculate_rewards(
                    self.reward_language_features, frames_embeddings
                )
                if isinstance(reward, th.Tensor):
                    reward = reward.detach().cpu().numpy().item()
                wandb.log({"train/learned_reward": reward})
                self.past_observations = []
                self.raw_observations = []
            else:
                reward = 0
        
        wandb_reward = reward
        reward /= self.reward_divisor
        if done:
            print(f"reward after divisor: {reward}")
        # Success bonus
        if info.get("success", False):
            reward += self.reward_model.success_bonus
            wandb_reward += self.reward_model.success_bonus
            print(f"train success reward: {reward}")
        if done:
            wandb.log({"train/learned_reward_with_success_bonus": wandb_reward})
        return obs, reward, done, info
    # @profile
    def reset(self):
        self.past_observations = []
        # print(len(self.raw_observations))
        self.raw_observations = []
        self.counter = 0

        obs = self.env.reset()

        # This is for the reward function
        image = self.env.render()
        image_for_model = image[None, None, :, :, :]
        # print(image_for_model.shape)
        encoded_image = self.image_encoder.encode_images(image_for_model).squeeze()

        if self.is_state_based is False:
            if self.use_proprio:
                proprio = obs[0:4]
                obs = np.concatenate([encoded_image, proprio])

            else:
                obs = encoded_image
        # self.past_observations.append(encoded_image)

        return obs


class VLC_GVL_RewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_model: BaseRewardModel,
        language_features_reward: str,  # raw text
        use_proprio: bool = False,
        is_state_based: bool = False,
    ):
        super(VLC_GVL_RewardWrapper, self).__init__(env)
        self.reward_model = reward_model
        self.use_proprio = use_proprio
        self.language_features = language_features_reward  # raw text

        # VLC and GVL needs raw image and text
        self.past_observations: List[np.ndarray] = []
        self.counter = 0

        self.reward_divisor = self.reward_model.reward_divisor
        self.reward_at_every_step = self.reward_model.reward_at_every_step

    def step(self, action):
        self.counter += 1
        obs, original_reward, done, info = self.env.step(action)

        frame = self.env.render()
        self.past_observations.append(frame)

        reward = 0.0
        if done or self.reward_at_every_step:
            video_frames = np.stack(self.past_observations, axis=0)
            reward = self.reward_model.calculate_rewards(video_frames, self.language_features)
            self.past_observations = []

        reward /= self.reward_divisor
        if info.get("success", False):
            reward += self.reward_model.success_bonus
        print(f"reward: {reward}")

        return obs, reward, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0

        obs = self.env.reset()
        
        # frame = self.env.render()
        # self.past_observations.append(frame)

        return obs

# Environment keeps an aggregate reward at each step and outputs it only when the episode ends
class RewardAtEndWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(RewardAtEndWrapper, self).__init__(env)
        # Keep track of the total reward
        self.total_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_reward += reward
        if done:
            final_reward = self.total_reward
            self.total_reward = 0  # 重置为下一个episode做准备
            return obs, final_reward, done, info
        else:
            return obs, 0, done, info  # 在episode未结束时返回0


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, divisor):
        super(RewardScaleWrapper, self).__init__(env)
        self.divisor = divisor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward / self.divisor, done, info
