import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from reward_model.base_reward_model import BaseRewardModel


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, reward_model: BaseRewardModel
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if "image_feature" in key:
                total_concat_size += subspace.shape[0]
            if "lang_feature" in key:
                total_concat_size += subspace.shape[0]

            elif "text" in key:
                # We will call the reward model's encode text function
                extractors[key] = lambda x: reward_model.encode_text(x)
                total_concat_size += reward_model.policy_text_output_dim
            if "image" in key:
                # We will call the reward model's encode video function
                extractors[key] = reward_model.encode_video
                total_concat_size += reward_model.img_output_dim

            elif key == "proprio":
                # Just append the proprio
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        # let's look through the keys. Assert that image and image_feature are not present at the same time
        # similarlty, assert that the text and lang_feature are not present at the same time

        # Order is proprio, text_feature, image_features

        image_feature_keys = [
            key for key in observations.keys() if "image_feature" in key
        ]
        image_keys = [
            key
            for key in observations.keys()
            if ("image" in key and "image_feature" not in key)
        ]

        # Sort the images keys in case there are multiple images
        image_keys = sorted(image_keys)
        image_feature_keys = sorted(image_feature_keys)

        text_feature_keys = [
            key for key in observations.keys() if "lang_feature" in key
        ]

        text_keys = [
            key
            for key in observations.keys()
            if ("text" in key and "lang_feature" not in key)
        ]

        assert len(image_feature_keys) > 0 and len(image_keys) > 0, (
            "Only one of image_feature or image should be present"
        )
        assert len(text_feature_keys) > 0 and len(text_keys) > 0, (
            "Only one of lang_feature or text should be present"
        )

        encoded_tensor_list = []

        # Add proprio if present
        if "proprio" in observations:
            encoded_tensor_list.append(
                self.extractors["proprio"](observations["proprio"])
            )

        # Add text features
        for key in text_keys:
            encoded_tensor_list.append(self.extractors[key](observations[key]))

        for key in text_feature_keys:
            encoded_tensor_list.append(self.extractors[key](observations[key]))

        # Add image features
        for key in image_keys:
            encoded_tensor_list.append(self.extractors[key](observations[key]))
        for key in image_feature_keys:
            encoded_tensor_list.append(self.extractors[key](observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
