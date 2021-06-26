from typing import Tuple, List, Type

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.autograd import Variable

from yacht.data.datasets import DayForecastDataset


class MultipleTimeFramesFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim,
            intervals: List[str],
            features: List[str],
            activation_fn: nn.Module,
            drop_out_p: float = 0.5,
            device='cuda'
    ):
        super().__init__(observation_space, features_dim)
        self.intervals = intervals
        self.features = features

        self.drop_out_layer = nn.Dropout(p=drop_out_p)
        self.relu_layer = activation_fn()
        self.layers = {
            'dense': dict(),
            'weight': dict()
        }
        for interval in self.intervals:
            bar_units = DayForecastDataset.get_day_bar_units_for(interval)
            self.layers['dense'][interval] = nn.Linear(
                bar_units,
                features_dim,
            ).to(device)
            self.layers['weight'][interval] = Variable(
                torch.tensor(1 / len(self.intervals), device=device),
                requires_grad=True
            ).to(device)
        self.global_feature_dense_layer = nn.Linear(
            len(self.features) * features_dim,
            features_dim
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.to_dict(observations)
        batch_size, window_size, _, _ = observations['1d'].shape

        features = []
        for interval in self.intervals:
            layer = self.layers['dense'][interval]
            weight = self.layers['weight'][interval]

            interval_feature = layer(observations[interval])
            interval_feature = self.relu_layer(interval_feature)
            interval_feature = self.drop_out_layer(interval_feature)
            interval_feature = interval_feature * weight

            features.append(interval_feature)

        features = torch.stack(features, dim=0)
        features = torch.sum(features, dim=0)
        features = torch.reshape(features, (batch_size, window_size, -1))

        features = self.global_feature_dense_layer(features)
        features = self.relu_layer(features)
        features = self.drop_out_layer(features)

        if window_size == 1:
            features = torch.squeeze(features, dim=1)

        return features

    def to_dict(self, observations: torch.Tensor) -> dict:
        batch_size = observations.shape[0]
        window_size = observations.shape[1]
        features_size = observations.shape[3]

        new_observation = dict()
        current_index = 0
        for interval in self.intervals:
            bar_units = DayForecastDataset.get_day_bar_units_for(interval)
            new_observation[interval] = observations[:, :, current_index:current_index + bar_units, :]
            new_observation[interval] = new_observation[interval].reshape(batch_size, window_size, -1, features_size)
            new_observation[interval] = torch.transpose(new_observation[interval], 2, 3)

            current_index += bar_units

        return new_observation


class DayForecastNetwork(nn.Module):
    def __init__(
            self,
            features_dim: int,
            window_size: int,
            intervals: List[str],
            activation_fn: Type[nn.Module],
    ):
        super().__init__()
        self.features_dim = features_dim
        self.window_size = window_size
        self.intervals = intervals

        self.activation_fn = activation_fn()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

        self.conv1 = nn.Conv2d(
            in_channels=features_dim,
            out_channels=features_dim,
            kernel_size=(1, len(self.intervals)),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()
        self.conv2 = nn.Conv2d(
            in_channels=features_dim,
            out_channels=features_dim * 2,
            kernel_size=(3, 1),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()

        self.conv_policy_net = nn.Conv2d(
            in_channels=features_dim * 2,
            out_channels=features_dim,
            kernel_size=(8, 1),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()

        self.conv_value_net = nn.Conv2d(
            in_channels=features_dim * 2,
            out_channels=features_dim,
            kernel_size=(8, 1),
            padding=(0, 0),
            stride=(1, 1)
        ).cuda()

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = self.conv1(features)
        features = self.activation_fn(features)
        features = self.conv2(features)
        features = self.activation_fn(features)

        policy_features = self.conv_policy_net(features)
        policy_features = self.activation_fn(policy_features)

        value_features = self.conv_value_net(features)
        value_features = self.activation_fn(value_features)

        batch_num = features.shape[0]
        policy_features = policy_features.reshape(batch_num, -1)
        value_features = value_features.reshape(batch_num, -1)

        return policy_features, value_features
