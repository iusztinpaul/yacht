from typing import Tuple, List, Type

import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from yacht.data.datasets import DayForecastDataset


class MultipleTimeFramesFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim, intervals: List[str]):
        super().__init__(observation_space, features_dim)
        self.intervals = intervals

        self.input_dense_layers = dict()
        for interval in self.intervals:
            bar_units = DayForecastDataset.get_day_bar_units_for(interval)
            self.input_dense_layers[interval] = nn.Linear(bar_units, features_dim).cuda()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = self.to_dict(observations)

        features = []
        for interval, layer in self.input_dense_layers.items():
            features.append(
                layer(observations[interval])
            )

        features = th.stack(features, dim=1)
        features = th.transpose(features, 1, 3)

        return features

    def to_dict(self, observations: th.Tensor) -> dict:
        new_observation = dict()
        current_index = 0
        for interval in self.intervals:
            # TODO: Check if this splitting is correct
            bar_units = DayForecastDataset.get_day_bar_units_for(interval)
            new_observation[interval] = observations[:, :, current_index:current_index + bar_units]
            current_index += bar_units

        return new_observation


class DayForecastNetwork(nn.Module):
    def __init__(
            self,
            features_dim: int,
            window_size: int,
            activation_fn: Type[nn.Module],
    ):
        super().__init__()
        self.features_dim = features_dim
        self.window_size = window_size

        self.activation_fn = activation_fn()
        self.latent_dim_pi = features_dim * 2 * window_size
        self.latent_dim_vf = features_dim * 2 * window_size

        self.conv1 = nn.Conv2d(
            in_channels=features_dim,
            out_channels=features_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()

        self.conv_policy_net = nn.Conv2d(
            in_channels=features_dim,
            out_channels=features_dim * 2,
            kernel_size=(1, 3),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()

        self.conv_value_net = nn.Conv2d(
            in_channels=features_dim,
            out_channels=features_dim * 2,
            kernel_size=(1, 3),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = self.conv1(features)
        features = self.activation_fn(features)

        policy_features = self.conv_policy_net(features)
        policy_features = self.activation_fn(policy_features)

        value_features = self.conv_value_net(features)
        value_features = self.activation_fn(value_features)

        batch_num = features.shape[0]
        policy_features = policy_features.reshape(batch_num, -1)
        value_features = value_features.reshape(batch_num, -1)

        return policy_features, value_features
