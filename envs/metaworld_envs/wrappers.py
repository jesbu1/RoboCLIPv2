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
import imageio
import os
from datetime import datetime
from gym.wrappers.normalize import NormalizeReward

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
        # Use absolute path
        self.video_dir = os.path.abspath("videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
            print(f"Created video directory at: {self.video_dir}")

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
        self.episode_counter = 0
        self.dense_eval = dense_eval
        self.total_success_bonus = 0

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
    def save_video(self, frames, reward):
        if not frames:
            print("No frames to save")
            return
            
        # Ensure frames are numpy arrays
        frames = [frame if isinstance(frame, np.ndarray) else frame.cpu().numpy() for frame in frames]
        
        # Ensure data type is uint8
        frames = [frame.astype(np.uint8) for frame in frames]
        
        # Ensure channel order is RGB
        frames = [frame[..., :3] for frame in frames]  # Only take the first 3 channels
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_dir, f"episode_{self.episode_counter}_{reward}.mp4")
        print(f"Attempting to save video to: {video_path}")
        try:
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Video successfully saved to: {video_path} with reward: {reward}")
        except Exception as e:
            print(f"Error saving video: {e}")
            print(f"Current working directory: {os.getcwd()}")

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
            if self.reward_model.name == "RewindRewardModel":
                stacked_sequence = np.stack(self.past_observations, axis=0)
                stacked_sequence = (
                    th.from_numpy(stacked_sequence).float().to(self.reward_model.device)
                ).unsqueeze(0)
            elif self.reward_model.name == "LIVRewardModel":
                stacked_sequence = th.from_numpy(self.reward_model.encode_images(
                    image_for_model
                )).unsqueeze(0)
                # print(f"stacked_sequence shape: {stacked_sequence.shape}") # (1, 1, 1024)

            reward = self.reward_model.calculate_rewards(
                self.reward_language_features, stacked_sequence
            )

            if isinstance(reward, th.Tensor):
                reward = reward.detach().cpu().numpy().item()
            # print(f"reward: {reward}")
            wandb.log({"train/learned_reward_per_step": reward})
            # print(f"reward : {reward}")
            # exit()
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
                # print(f"reward: {reward}")
                # exit()
                if self.episode_counter % 350 == 0:
                    # Convert raw_observations to numpy array and save as video
                    frames_np = [frame.squeeze() for frame in self.raw_observations]
                    self.save_video(frames_np, reward)
                if isinstance(reward, th.Tensor):
                    reward = reward.detach().cpu().numpy().item()
                
                self.past_observations = []
                self.raw_observations = []
            else:
                reward = 0

        wandb_reward = reward
        reward /= self.reward_divisor
        if done:
            # print(f"reward after divisor: {reward}")
            self.episode_counter += 1
            wandb.log({"train/learned_reward": wandb_reward})

        # Normalize reward
        # if self.counter == 1:
        #     self.offset = reward
        #     reward -= self.offset
        # elif self.counter > 1:
        #     reward -= self.offset

        # Success bonus
        if info.get("success", False):
            reward += self.reward_model.success_bonus
            wandb_reward += self.reward_model.success_bonus
            self.total_success_bonus += self.reward_model.success_bonus
            print(f"The {self.episode_counter}th episode {self.counter}th step, train success reward: {reward}")
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
        self.past_observations.append(encoded_image)
        self.raw_observations.append(image_for_model)
        wandb.log({"train/total_success_bonus": self.total_success_bonus})
        self.total_success_bonus = 0
        return obs


class VLC_GVL_RewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_model: BaseRewardModel,
        image_encoder,
        language_features_reward: str,
        use_proprio: bool = False,
        is_state_based: bool = False,
        dense_eval: bool = False,
    ):
        super(VLC_GVL_RewardWrapper, self).__init__(env)
        self.reward_model = reward_model
        self.image_encoder = image_encoder
        self.use_proprio = use_proprio
        self.language_features = language_features_reward
        self.is_state_based = is_state_based
        self.dense_eval = dense_eval
        # VLC and GVL needs raw image and text
        self.past_observations: List[np.ndarray] = []
        self.counter = 0
        self.raw_observations = []
        self.episode_counter = 0
        self.total_success_bonus = 0
        self.reward_divisor = self.reward_model.reward_divisor
        self.reward_at_every_step = self.reward_model.reward_at_every_step
        
        if self.is_state_based is False:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.image_encoder.img_output_dim + (4 if self.use_proprio else 0),
                ),
                dtype=np.float32,
            )

        # =========  视频处理参数  ========= #
        # 允许从 reward_model 继承 max_frames，否则默认 32
        self.max_frames: int = getattr(self.reward_model, "max_frames", 12)
        # 若帧数不足也要拉伸到 max_frames
        self.stretch_partial_videos: bool = True

    def _padding_frames(self, video_np: "np.ndarray") -> "np.ndarray":
        """According to max_frames, padding/sampling frames of video, directly operate on numpy, keep [T,H,W,C] layout."""
        import numpy as np

        num_frames = video_np.shape[0]

        if num_frames >= self.max_frames or self.stretch_partial_videos:
            float_indices = np.linspace(0, num_frames - 1, self.max_frames)
            indices = np.round(float_indices).astype(int)
        else:
            indices = np.arange(num_frames)

        # Optional custom frame sequence
        if hasattr(self, "frame_indices_to_use") and self.frame_indices_to_use is not None:
            if len(indices) > len(self.frame_indices_to_use):
                indices = indices[self.frame_indices_to_use]

        return video_np[indices]

    def step(self, action):
        self.counter += 1
        obs, original_reward, done, info = self.env.step(action)
        proprio = obs[0:4]
        frame = self.env.render()
        self.raw_observations.append(frame)
        # Input should be of shape (batch_size, num_frames, height, width, channels)
        # However, the input is of shape (height, width, channels)
        image_for_model = frame[None, None, :, :, :]
        encoded_image = self.image_encoder.encode_images(
            image_for_model
        ).squeeze()
        obs = encoded_image
        if self.use_proprio:
                obs = np.concatenate([obs, proprio])

        if self.dense_eval:
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

        if done or self.reward_at_every_step:
            video_frames = np.stack(self.raw_observations, axis=0)

            # ---------  Padding/sampling frames of video  --------- #
            video_frames = self._padding_frames(video_frames)
            # print(f"video_frames shape: {video_frames.shape}")
            reward = self.reward_model.calculate_rewards(video_frames, self.language_features)
            if self.counter == 1:
                self.offset = reward
            reward -= self.offset
            if done:
                wandb.log({"train/learned_reward": reward})
            else:
                wandb.log({"train/learned_reward_per_step": reward})

        reward /= self.reward_divisor
        if info.get("success", False):
            # reward += self.reward_model.success_bonus
            self.total_success_bonus += self.reward_model.success_bonus
            print(f"The {self.episode_counter}th episode {self.counter}th step, train success reward: {reward}")
        if done:
            self.episode_counter += 1
            # wandb.log({"train/learned_reward_with_success_bonus": reward})

        return obs, reward, done, info

    def reset(self):
        self.raw_observations = []
        self.counter = 0
        wandb.log({"train/total_success_bonus": self.total_success_bonus})
        self.total_success_bonus = 0
        obs = self.env.reset()
        proprio = obs[0:4]
        frame = self.env.render()
        image_for_model = frame[None, None, :, :, :]
        encoded_image = self.image_encoder.encode_images(image_for_model).squeeze()

        if self.is_state_based is False:
            if self.use_proprio:
                proprio = obs[0:4]
                obs = np.concatenate([encoded_image, proprio])

            else:
                obs = encoded_image

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
            self.total_reward = 0  # Reset for the next episode
            return obs, final_reward, done, info
        else:
            return obs, 0, done, info  # Return 0 when episode is not finished


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, divisor):
        super(RewardScaleWrapper, self).__init__(env)
        self.divisor = divisor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward / self.divisor, done, info

class RecordRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super(RecordRewardWrapper, self).__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.reward_model.reward_at_every_step:
            wandb.log({"train/normalized_reward_per_step": reward})
        if done:
            wandb.log({"train/normalized_reward": reward})
        if info.get("success", False):
            reward += self.reward_model.success_bonus
        if done:
            wandb.log({"train/normalized_reward_with_success_bonus": reward})
        
        return obs, reward, done, info


class RunningMeanStd:
    """Running mean & std, numpy 版（与 OpenAI Baselines 一致）"""
    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """x: (...,) 任意维度 batch"""
        batch_mean = np.mean(x, axis=0)
        batch_var  = np.var(x,  axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta      = batch_mean - self.mean
        tot_count  = self.count + batch_count

        new_mean   = self.mean + delta * batch_count / tot_count
        m_a        = self.var * self.count
        m_b        = batch_var * batch_count
        M2         = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var    = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count


class RewardNormalize(gym.Wrapper):

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.gamma   = gamma
        self.epsilon = epsilon
        self.num_envs       = getattr(env, "num_envs", 1)
        self.is_vector_env  = getattr(env, "is_vector_env", False)
        self.return_rms     = RunningMeanStd(shape=()) 
        self.returns        = np.zeros(self.num_envs, dtype=np.float64)

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)
        if not self.is_vector_env:
            rewards = np.array([rewards], dtype=np.float64)
            dones   = np.array([dones],   dtype=np.float64) 

        self.returns = self.returns * self.gamma * (1.0 - dones) + rewards
        self.return_rms.update(self.returns)

        norm_rews = rewards / np.sqrt(self.return_rms.var + self.epsilon)

        if not self.is_vector_env:
            norm_rews = norm_rews[0]
        return obs, norm_rews, dones if self.is_vector_env else bool(dones[0]), infos

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
