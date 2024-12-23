import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gym import spaces

from reward_model.base_reward_model import BaseRewardModel


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
        t = self.counter / 500  # Assuming max steps is 500
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
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return np.concatenate([obs, self.language_features])


class LearnedRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_model: BaseRewardModel,
        language_features: th.Tensor,
        is_state_based: bool = False,
        dense_eval: bool = False,
    ):
        super(LearnedRewardWrapper, self).__init__(env)
        self.reward_model = reward_model
        self.is_state_based = is_state_based

        if self.is_state_based is False:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.reward_model.img_output_dim,),
                dtype=np.float32,
            )

        self.past_observations = []
        self.counter = 0

        self.dense_eval = dense_eval

        self.reward_at_every_step = self.reward_model.reward_at_every_step
        self.reward_language_features = (
            th.Tensor(language_features)
            .float()
            .to(self.reward_model.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def step(self, action):
        self.counter += 1
        obs, original_reward, done, info = self.env.step(action)

        encoded_image = None
        # IF the model is state-based and is dense/sparse reward, we can skip this
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
                encoded_image = self.reward_model.encode_images(
                    image_for_model
                ).squeeze()

        if self.is_state_based is False:
            # obs = np.concatenate([obs, self.reward_model(obs)])
            obs = encoded_image

        if self.reward_model.name == "dense" or self.dense_eval:
            return obs, original_reward, done, info
        # Check if this is sparse/dense reward
        elif self.reward_model.name == "sparse":
            sparse_reward = (
                self.reward_model.success_bonus if info.get("success", False) else 0.0
            )
            reward = sparse_reward
            return obs, reward, done, info

        if encoded_image is not None:
            self.past_observations.append(encoded_image)

        if self.reward_at_every_step:
            stacked_sequence = np.stack(self.past_observations, axis=1)
            stacked_sequence = (
                th.from_numpy(stacked_sequence).float().to(self.reward_model.device)
            )

            reward = self.reward_model.calculate_rewards(
                self.reward_language_features, stacked_sequence
            )

        else:
            if done:
                stacked_sequence = np.stack(self.past_observations, axis=0)
                stacked_sequence = (
                    th.from_numpy(stacked_sequence).float().to(self.reward_model.device)
                )

                reward = self.reward_model.calculate_rewards(
                    self.reward_language_features, stacked_sequence.unsqueeze(0)
                )
                self.past_observations = []
            else:
                reward = 0

        # Success bonus
        if info.get("success", False):
            reward += self.reward_model.success_bonus

        return obs, reward, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0

        obs = self.env.reset()

        # This is for the reward function
        image = self.env.render()
        image_for_model = image[None, None, :, :, :]
        encoded_image = self.reward_model.encode_images(image_for_model).squeeze()

        if self.is_state_based is False:
            obs = encoded_image

        self.past_observations.append(encoded_image)

        return obs


# Wrapper for similarity-based observations
class SimilarityRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        target_embedding,
        transform_model,
        video_processor,
        pca_video_model=None,
        max_sim=None,
        succ_bonus=0,
        time_penalty=0.1,
        time_reward=1.0,
        threshold_reward=False,
        norm_output=True,
        baseline=False,
        project_reward=False,
    ):
        super(SimilarityRewardWrapper, self).__init__(env)
        self.target_embedding = target_embedding  # Target video embedding
        self.transform_model = transform_model  # Transformation model (if any)
        self.video_processor = (
            video_processor  # Video frame processor (for neural network input)
        )
        self.pca_video_model = (
            pca_video_model  # Optional PCA model for dimensionality reduction
        )
        self.max_sim = max_sim  # Maximum similarity threshold for rewards
        self.succ_bonus = succ_bonus  # Bonus if the task is successfully completed
        self.time_penalty = time_penalty  # Penalty based on time spent
        self.time_reward = time_reward  # Scaling factor for reward
        self.threshold_reward = threshold_reward  # Use thresholded reward logic
        self.norm_output = norm_output  # Normalize output embeddings
        self.baseline = baseline  # Use baseline behavior
        self.project_reward = (
            project_reward  # Option to project rewards based on similarity
        )
        self.past_observations = []  # Storage for past observations (frames)
        self.counter = 0  # Step counter to track time

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        # render every 10 steps
        if self.counter % 10 == 0:
            self.past_observations.append(
                self.env.render()
            )  # Collect frame from the environment

        self.counter += 1

        if done:
            with th.no_grad():
                # Process the video frames into embeddings
                frames = self.video_processor.adjust_frames(self.past_observations)
                video_embedding = self.transform_model.get_video_features(frames)

                # Normalize the embeddings if required
                if self.norm_output:
                    video_embedding = normalize_embeddings(video_embedding).float()
                    self.target_embedding = normalize_embeddings(
                        self.target_embedding
                    ).float()

                # Apply PCA transformation if required
                if self.pca_video_model is not None:
                    video_embedding = (
                        th.from_numpy(
                            self.pca_video_model.transform(video_embedding.cpu())
                        )
                        .float()
                        .cuda()
                    )

                # Further transform the embeddings if using non-baseline behavior
                if not self.baseline:
                    video_embedding = self.transform_model(video_embedding)

                # Normalize embeddings again
                video_embedding = normalize_embeddings(video_embedding).float()

                # Calculate similarity between target and current video embedding
                similarity_matrix = th.matmul(
                    self.target_embedding, video_embedding.t()
                )
                reward = similarity_matrix.cpu().numpy()[0][0]

                # Scale reward by time
                reward *= self.time_reward

                # Thresholded reward logic
                if self.threshold_reward:
                    if self.max_sim is not None:
                        if reward < self.max_sim:
                            reward = 0.0
                        elif self.project_reward:
                            reward = (
                                (reward - self.max_sim) / (100 - self.max_sim) * 100
                            )
                    else:
                        raise ValueError(
                            "Max similarity score must be provided for thresholded reward."
                        )

                # Add bonus if task was successful
                if info.get("success", False):
                    reward += self.succ_bonus

                # Apply time penalty
                reward -= self.time_penalty

            return obs, reward, done, info

        # Default time penalty if not finished
        return obs, -self.time_penalty, done, info

    def reset(self):
        self.past_observations = []  # Clear observation history on reset
        self.counter = 0  # Reset time counter
        return self.env.reset()
