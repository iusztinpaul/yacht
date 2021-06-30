from typing import Tuple, List, Type

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.autograd import Variable

from yacht.agents.utils import unflatten_observations
from yacht.data.datasets import DayForecastDataset


class MultipleTimeFramesFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: List[int],
            window_size: int,
            intervals: List[str],
            features: List[str],
            env_features_len: int,
            activation_fn: nn.Module,
            drop_out_p: float = 0.5
    ):
        super().__init__(observation_space, features_dim[0])

        assert len(features_dim) == 2

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len

        self.interval_encoders = dict()
        for interval in self.intervals:
            bar_units = DayForecastDataset.get_day_bar_units_for(interval)
            self.interval_encoders[interval] = IntervalObservationEncoder(
                num_input_channel=len(self.features),
                num_output_channel=features_dim[0],
                kernel_size=self.window_size * bar_units,
                initial_output_weight_value=1 / len(self.intervals)
            )

        self.interval_encoders = nn.ModuleDict(self.interval_encoders)

        self.drop_out_layer = nn.Dropout(p=drop_out_p)
        self.relu_layer = activation_fn()
        self.dense_layer = nn.Linear(
            len(self.intervals) * features_dim[0] + self.env_features_len,
            features_dim[1]
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(observations, self.intervals)
        batch_size, window_size, channel_size, _ = observations['1d'].shape

        features = []
        for interval in self.intervals:
            encoder = self.interval_encoders[interval]
            interval_feature = encoder(observations[interval])

            features.append(interval_feature)

        # Add env_features after learning the price window features.
        features.append(observations['env_features'])
        features = torch.cat(features, dim=-1)

        features = self.dense_layer(features)
        features = self.relu_layer(features)
        features = self.drop_out_layer(features)

        return features


class IntervalObservationEncoder(nn.Module):
    def __init__(
            self,
            num_input_channel: int,
            num_output_channel: int,
            kernel_size: int,
            initial_output_weight_value: float
    ):
        super().__init__()

        assert initial_output_weight_value <= 1

        self.conv_1d = nn.Conv1d(
            in_channels=num_input_channel,
            out_channels=num_output_channel,
            kernel_size=kernel_size,
            stride=1
        )
        self.weight = nn.Parameter(
            torch.tensor(initial_output_weight_value).type(torch.FloatTensor),
            requires_grad=True,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, channel_size, _ = observation.shape

        interval_feature = torch.transpose(observation, 1, 2)
        interval_feature = interval_feature.reshape(batch_size, channel_size, -1)
        interval_feature = self.conv_1d(interval_feature)
        interval_feature = torch.squeeze(interval_feature, dim=-1)
        interval_feature = interval_feature * self.weight

        return interval_feature


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
