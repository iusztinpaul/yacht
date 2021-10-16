from typing import List

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from yacht.agents.misc import unflatten_observations


class RecurrentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: List[int],
            window_size: int,
            intervals: List[str],
            features: List[str],
            env_features_len: int,
            num_assets: int,
            activation_fn: nn.Module,
            rnn_layer_type: nn.Module,
            drop_out_p: float = 0.5
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.num_rnn_layers = len(features_dim[1:-1])

        self.public_mlp = nn.Sequential(
            nn.Linear(in_features=len(self.features) * self.num_assets, out_features=features_dim[0]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )
        self.public_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=drop_out_p
        )
        self.public_dropout = nn.Dropout(p=drop_out_p)

        self.private_mlp = nn.Sequential(
            nn.Linear(in_features=env_features_len, out_features=features_dim[0]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )
        self.private_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=drop_out_p
        )
        self.private_dropout = nn.Dropout(p=drop_out_p)

        self.output_mlp = nn.Sequential(
            nn.Linear(features_dim[1] * 2, features_dim[-1]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.env_features_len,
            num_assets=self.num_assets
        )
        batch_size, window_size, bar_size, num_assets_size, features_size = observations['1d'].shape
        public_input = observations['1d']
        public_input = public_input.reshape(batch_size, window_size, -1)

        batch_size, window_size, env_features = observations['env_features'].shape
        private_input = observations['env_features']

        public_input = self.public_mlp(public_input)
        public_input, _ = self.public_recurrent(public_input)
        public_input = public_input[:, -1, :]
        public_input = self.public_dropout(public_input)

        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input[:, -1, :]
        private_input = self.private_dropout(private_input)

        output = torch.cat([public_input, private_input], dim=-1)
        output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output


class RecurrentAttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: List[int],
            window_size: int,
            intervals: List[str],
            features: List[str],
            env_features_len: int,
            num_assets: int,
            activation_fn: nn.Module,
            rnn_layer_type: nn.Module,
            drop_out_p: float = 0.5
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.num_rnn_layers = len(features_dim[1:-1])

        self.public_mlp = nn.Sequential(
            nn.Linear(in_features=len(self.features) * self.num_assets, out_features=features_dim[0]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )
        self.public_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=drop_out_p
        )
        self.public_attention = nn.MultiheadAttention(
            embed_dim=features_dim[1],
            num_heads=features_dim[1] // 8,
            dropout=drop_out_p
        )

        self.private_mlp = nn.Sequential(
            nn.Linear(in_features=env_features_len, out_features=features_dim[0]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )
        self.private_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=drop_out_p
        )
        self.private_attention = nn.MultiheadAttention(
            embed_dim=features_dim[1],
            num_heads=features_dim[1] // 8,
            dropout=drop_out_p
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(features_dim[1] * 2, features_dim[-1]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.env_features_len,
            num_assets=self.num_assets
        )
        batch_size, window_size, bar_size, num_assets_size, features_size = observations['1d'].shape
        public_input = observations['1d']
        public_input = public_input.reshape(batch_size, window_size, -1)

        batch_size, window_size, env_features = observations['env_features'].shape
        private_input = observations['env_features']

        public_input = self.public_mlp(public_input)
        public_input, _ = self.public_recurrent(public_input)
        public_input = public_input.transpose(0, 1)
        public_input, _ = self.public_attention(public_input, public_input, public_input)
        public_input = public_input[-1, :, :]

        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input.transpose(0, 1)
        private_input, _ = self.private_attention(private_input, private_input, private_input)
        private_input = private_input[-1, :, :]

        output = torch.cat([public_input, private_input], dim=-1)
        output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output


class RecurrentNPeriodsFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: List[int],
            window_size: int,
            num_periods: int,
            intervals: List[str],
            features: List[str],
            env_features_len: int,
            num_assets: int,
            activation_fn: nn.Module,
            rnn_layer_type: nn.Module,
            drop_out_p: float = 0.5
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'
        assert window_size % num_periods == 0

        self.window_size = window_size
        self.num_periods = num_periods
        self.period_length = self.window_size // self.num_periods
        assert self.period_length >= 3

        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.num_rnn_layers = len(features_dim[1:-1])

        self.public_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=len(self.features) * self.num_assets,
                out_channels=2 * len(self.features) * self.num_assets,
                kernel_size=(1,)
            ),
            activation_fn(),
            nn.Conv1d(
                in_channels=2 * len(self.features) * self.num_assets,
                out_channels=4 * len(self.features) * self.num_assets,
                kernel_size=(3,)
            ),
            activation_fn()
        )
        self.public_mlp = nn.Sequential(
            nn.Linear(
                in_features=4 * len(self.features) * self.num_assets * (self.period_length - 3 + 1),
                out_features=features_dim[0]
            ),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )
        self.public_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=drop_out_p
        )
        self.public_dropout = nn.Dropout(p=drop_out_p)

        self.private_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.env_features_len,
                out_channels=2 * self.env_features_len,
                kernel_size=(1,)
            ),
            activation_fn()
        )
        self.private_mlp = nn.Sequential(
            nn.Linear(
                in_features=2 * self.env_features_len * self.period_length,
                out_features=features_dim[0]
            ),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )
        self.private_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=drop_out_p
        )
        self.private_dropout = nn.Dropout(p=drop_out_p)

        self.output_mlp = nn.Sequential(
            nn.Linear(features_dim[1] * 2, features_dim[-1]),
            activation_fn(),
            nn.Dropout(p=drop_out_p)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.env_features_len,
            num_assets=self.num_assets
        )
        # Reshape public variables.
        batch_size, window_size, bar_size, num_assets_size, features_size = observations['1d'].shape
        public_input = observations['1d']
        public_input = public_input.reshape(
            batch_size * self.num_periods,
            self.period_length,
            num_assets_size * features_size
        )
        public_input = public_input.transpose(1, 2)

        # Reshape private variables.
        batch_size, window_size, env_features = observations['env_features'].shape
        private_input = observations['env_features']
        private_input = private_input.reshape(
            batch_size * self.num_periods,
            self.period_length,
            env_features
        )
        private_input = private_input.transpose(1, 2)

        # Forward public input.
        public_input = self.public_cnn(public_input)
        public_input = public_input.reshape(batch_size, self.num_periods, -1)
        public_input = self.public_mlp(public_input)
        public_input, _ = self.public_recurrent(public_input)
        public_input = public_input[:, -1, :]
        public_input = self.public_dropout(public_input)

        # Forward private input.
        private_input = self.private_cnn(private_input)
        private_input = private_input.reshape(batch_size, self.num_periods, -1)
        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input[:, -1, :]
        private_input = self.private_dropout(private_input)

        # Forward shared features.
        output = torch.cat([public_input, private_input], dim=-1)
        output = self.output_mlp(output)

        return output
