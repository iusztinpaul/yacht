from typing import List

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from yacht.agents.misc import unflatten_observations
from yacht.data.datasets import DayMultiFrequencyDataset


class MultiFrequencyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: List[int],
            window_size: int,
            intervals: List[str],
            features: List[str],
            num_assets: int,
            include_weekends: bool,
            env_features_len: int,
            activation_fn: nn.Module,
            drop_out_p: float = 0.5
    ):
        super().__init__(observation_space, features_dim[1])

        assert len(features_dim) == 2

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.num_assets = num_assets
        self.include_weekends = include_weekends
        self.env_features_len = env_features_len

        self.interval_encoders = dict()
        for interval in self.intervals:
            bar_units = DayMultiFrequencyDataset.get_day_bar_units_for(interval, include_weekends=include_weekends)
            self.interval_encoders[interval] = IntervalObservationEncoder(
                num_input_channel=len(self.features) + self.env_features_len,
                num_output_channel=features_dim[0],
                kernel_size=self.window_size * bar_units,
                initial_output_weight_value=1 / len(self.intervals)
            )

        self.interval_encoders = nn.ModuleDict(self.interval_encoders)

        self.drop_out_layer = nn.Dropout(p=drop_out_p)
        self.relu_layer = activation_fn()
        self.dense_layer = nn.Linear(
            len(self.intervals) * features_dim[0],
            features_dim[1]
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(observations, self.intervals, self.env_features_len, self.num_assets)
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


class MultiFrequencyRecurrentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: List[int],
            window_size: int,
            intervals: List[str],
            features: List[str],
            env_features_len: int,
            num_assets: int,
            include_weekends: bool,
            activation_fn: nn.Module,
            rnn_layer_type: nn.Module,
            aggregation_type: str = 'prod'
    ):
        assert aggregation_type in ('concat', 'sum', 'prod', 'max')

        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.include_weekends = include_weekends
        self.aggregation_type = aggregation_type
        self.num_rnn_layers = len(features_dim[1:-1])

        self.public_modules = nn.ModuleDict()
        for interval in self.intervals:
            num_bars = DayMultiFrequencyDataset.get_day_bar_units_for(interval, include_weekends=include_weekends)
            self.public_modules[interval] = nn.ModuleDict()
            self.public_modules[interval]['mlp'] = nn.Sequential(
                nn.Linear(in_features=len(self.features) * self.num_assets * num_bars, out_features=features_dim[0]),
                activation_fn()
            )
            self.public_modules[interval]['rnn'] = rnn_layer_type(
                features_dim[0],
                features_dim[1],
                num_layers=self.num_rnn_layers,
                batch_first=True
            )

        self.private_mlp = nn.Sequential(
            nn.Linear(in_features=env_features_len, out_features=features_dim[0]),
            activation_fn()
        )
        self.private_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )

        if self.aggregation_type == 'concat':
            num_in_features = features_dim[1] * (1 + len(self.intervals))
        else:
            num_in_features = features_dim[1] * 2
        self.output_mlp = nn.Sequential(
            nn.Linear(num_in_features, features_dim[-1]),
            activation_fn()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.env_features_len,
            num_assets=self.num_assets,
            include_weekends=self.include_weekends
        )
        batch_size, window_size, _, num_assets_size, features_size = observations['1d'].shape

        aggregated_public_features = []
        for interval in self.intervals:
            public_features = observations[interval]
            public_features = public_features.reshape(batch_size, window_size, -1)
            public_features = self.public_modules[interval]['mlp'](public_features)
            public_features, _ = self.public_modules[interval]['rnn'](public_features)
            public_features = public_features[:, -1, :]

            aggregated_public_features.append(public_features)

        if self.aggregation_type == 'concat':
            aggregated_public_features = torch.cat(aggregated_public_features, dim=-1)
        elif self.aggregation_type == 'sum':
            aggregated_public_features = torch.stack(aggregated_public_features, dim=1)
            aggregated_public_features = torch.sum(aggregated_public_features, dim=1)
        elif self.aggregation_type == 'prod':
            aggregated_public_features = torch.stack(aggregated_public_features, dim=1)
            aggregated_public_features = torch.prod(aggregated_public_features, dim=1)
        elif self.aggregation_type == 'max':
            aggregated_public_features = torch.stack(aggregated_public_features, dim=1)
            aggregated_public_features = torch.max(aggregated_public_features, dim=1).values

        private_input = observations['env_features']
        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input[:, -1, :]

        output = torch.cat([aggregated_public_features, private_input], dim=-1)
        output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output