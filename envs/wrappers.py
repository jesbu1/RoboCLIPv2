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
        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.env.observation_space.shape[0] + len(self.language_features),),
        #     dtype=np.float32,
        # )

        # The observation space is a dict
        # Let us add language_feature to the observation space
        current_obs_space = self.env.observation_space
        assert isinstance(current_obs_space, spaces.Dict), (
            "Observation space must be a Dict."
        )

        new_spaces = current_obs_space.spaces.copy()
        new_spaces["language_feature"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.language_features),),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(new_spaces)

    def _observation(self, observation):
        observation["language_feature"] = self.language_features
        return observation

    def reset(self):
        obs = self.env.reset()
        obs = self._observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info


class ImageEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super(ImageEmbeddingWrapper, self).__init__(env)
        self.reward_model = reward_model

        # The observation space is a dict
        # Let us add image_feature to the observation space

        current_obs_space = self.env.observation_space
        assert isinstance(current_obs_space, spaces.Dict), (
            "Observation space must be a Dict."
        )

        image_keys = self.env.image_keys

        # Define the new observation space
        new_spaces = current_obs_space.spaces.copy()
        for i, key in enumerate(image_keys):
            # Add a new key for the image feature corresponding to each image key
            new_spaces[f"image_feature_{i}"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(reward_model.img_output_dim,),
                dtype=np.float32,
            )

        # Set the updated observation space
        self.observation_space = spaces.Dict(new_spaces)

    def _observation(self, observation):
        for i, key in enumerate(self.image_keys):
            image = observation[key]
            image = image[None, None, :, :, :]
            image_feature = self.reward_model.encode_images(image).squeeze()
            observation[f"image_feature_{i}"] = image_feature

        return observation

    def reset(self):
        obs = self.env.reset()
        obs = self._observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info


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

        self.past_observations = []
        self.counter = 0

        self.dense_eval = dense_eval

        self.reward_at_every_step = self.reward_model.reward_at_every_step
        self.reward_divisor = self.reward_model.reward_divisor

        if language_features is not None:
            self.reward_language_features = (
                th.Tensor(language_features)
                .float()
                .to(self.reward_model.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            print("Language features are not provided in the reward model")
            print(
                "This may be valid if the user is using sparse/dense reward in a single task"
            )

    def step(self, action):
        self.counter += 1
        obs, original_reward, done, info = self.env.step(action)

        encoded_image = None
        # IF the model is state-based and is dense/sparse reward, we can skip this

        if f"image_feature_{self.image_reward_idx}" in obs:
            encoded_image = obs[f"image_feature_{self.image_reward_idx}"]

        if self.reward_model.name == "dense" or self.dense_eval:
            reward = original_reward / self.reward_divisor

            if info.get("success", False):
                reward += self.reward_model.success_bonus

            return obs, reward, done, info
        # Check if this is sparse/dense reward
        elif self.reward_model.name == "sparse":
            sparse_reward = (
                self.reward_model.success_bonus if info.get("success", False) else 0.0
            )
            # Note: No reward divisor for sparse reward.

            return obs, sparse_reward, done, info

        if encoded_image is not None:
            self.past_observations.append(encoded_image)

        assert self.reward_language_features is not None, (
            "Language features are None in the reward model"
        )

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

        reward /= self.reward_divisor

        # Success bonus
        if info.get("success", False):
            reward += self.reward_model.success_bonus

        return obs, reward, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0

        obs = self.env.reset()

        encoded_image = None

        if f"image_feature_{self.image_reward_idx}" in obs:
            encoded_image = obs[f"image_feature_{self.image_reward_idx}"]

        self.past_observations.append(encoded_image)

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
            self.total_reward = 0
            return obs, self.total_reward, done, info
        else:
            return obs, reward, done, info


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, divisor):
        super(RewardScaleWrapper, self).__init__(env)
        self.divisor = divisor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward / self.divisor, done, info


class ActionChunkingWrapper(gym.Wrapper):
    def __init__(self, env, chunk_size=15):
        super(ActionChunkingWrapper, self).__init__(env)
        self.chunk_size = chunk_size

        self.chunk = []

    def step(self, chunked_action: np.ndarray):
        # Unpack action

        if chunked_action is not None and chunked_action.ndim == 1:
            print("**" * 10)
            print()
            print("THE ACTION IS NOT CHUNKED")
            print("This may be okay if random exploration from SB3 is used")
            print()
            print("**" * 10)
            obs, reward, done, info = self.env.step(chunked_action)
            info["action"] = chunked_action[None, :]
            return obs, reward, done, info

        if self.is_chunk_empty:
            # Then let the action replace the chunk
            self.chunk = chunked_action
        else:
            # If chunk is not empty, we will assert that chunked_action is None
            assert chunked_action is None

        popped_action = self.chunk[0]
        self.chunk = self.chunk[1:]

        obs, reward, done, info = self.env.step(popped_action)

        info["action"] = popped_action

        return obs, reward, done, info

    @property
    def is_chunk_empty(self):
        return len(self.chunk) == 0

    def reset(self):
        self.chunk = []
        return self.env.reset()
