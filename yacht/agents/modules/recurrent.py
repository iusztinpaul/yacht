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
            nn.Linear(features_dim[1] * 2 * self.window_size, features_dim[-1]),
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
        public_input = self.public_dropout(public_input)

        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = self.private_dropout(private_input)

        output = torch.cat([public_input, private_input], dim=-1)
        output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output
