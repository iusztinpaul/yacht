from typing import List, Optional, Dict

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from yacht.agents.misc import unflatten_observations
from yacht.agents.modules.torch_layers import SimplifiedVariableSelectionNetwork, LinearStack
from yacht.data.datasets import DayMultiFrequencyDataset


class DayRecurrentFeatureExtractor(BaseFeaturesExtractor):
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
            rnn_layer_type: nn.Module
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.include_weekends = include_weekends
        self.num_rnn_layers = len(features_dim[1:-1])

        self.public_mlp = nn.Sequential(
            nn.Linear(in_features=len(self.features) * self.num_assets, out_features=features_dim[0]),
            activation_fn()
        )
        self.public_recurrent = rnn_layer_type(
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

        self.output_mlp = nn.Sequential(
            nn.Linear(features_dim[1] * 2, features_dim[-1]),
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
        batch_size, window_size, bar_size, num_assets_size, features_size = observations['1d'].shape
        public_input = observations['1d']
        public_input = public_input.reshape(batch_size, window_size, -1)

        batch_size, window_size, env_features = observations['env_features'].shape
        private_input = observations['env_features']

        public_input = self.public_mlp(public_input)
        public_input, _ = self.public_recurrent(public_input)
        public_input = public_input[:, -1, :]

        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input[:, -1, :]

        output = torch.cat([public_input, private_input], dim=-1)
        output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output


class DayVSNRecurrentFeatureExtractor(BaseFeaturesExtractor):
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
            dropout: Optional[float] = None
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.include_weekends = include_weekends
        self.num_rnn_layers = len(features_dim[1:-1])
        self.dropout = dropout

        self.public_vsn = SimplifiedVariableSelectionNetwork(
            public_features_len=len(self.features),
            private_features_len=None,
            num_assets=self.num_assets,
            hidden_size=features_dim[0],
            activation_fn=activation_fn,
            dropout=self.dropout
        )
        self.public_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )

        self.private_mlp = LinearStack(
            in_features=env_features_len,
            out_features=features_dim[0],
            activation_fn=activation_fn,
            n=1,
            hidden_features=features_dim[0],
            dropout=self.dropout
        )
        self.private_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )

        self.output_mlp = LinearStack(
            in_features=features_dim[1] * 2,
            out_features=features_dim[2],
            activation_fn=activation_fn,
            n=1,
            hidden_features=features_dim[2],
            dropout=self.dropout
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.env_features_len,
            num_assets=self.num_assets,
            include_weekends=self.include_weekends
        )
        batch_size, window_size, env_features = observations['env_features'].shape
        private_input = observations.pop('env_features')

        public_variables = self.split_variables(observations)
        public_input = self.public_vsn(public_variables)
        public_input, _ = self.public_recurrent(public_input)
        public_input = public_input[:, -1, :]

        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input[:, -1, :]

        output = torch.cat([public_input, private_input], dim=-1)
        output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output

    @classmethod
    def split_variables(cls, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        variables = dict()
        for key, value in observations.items():
            if key == 'env_features':
                variables['private_features'] = value
            elif key == '1d':
                num_assets = value.shape[3]
                for asset_idx in range(num_assets):
                    asset_values = value[..., asset_idx, :]
                    batch_size, window_size = asset_values.shape[:2]
                    asset_values = asset_values.view(batch_size, window_size, -1)
                    variables[f'public_features_{asset_idx}'] = asset_values
            else:
                raise RuntimeError(f'Unsupported observation type: {key}')

        return variables


class OnlyVSNRecurrentFeatureExtractor(BaseFeaturesExtractor):
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
            dropout: Optional[float] = None
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets
        self.include_weekends = include_weekends
        self.num_rnn_layers = len(features_dim[1:-1])
        self.dropout = dropout

        self.vsn = SimplifiedVariableSelectionNetwork(
            public_features_len=len(self.features),
            private_features_len=self.env_features_len,
            num_assets=self.num_assets,
            hidden_size=features_dim[0],
            activation_fn=activation_fn,
            dropout=self.dropout
        )
        self.recurrent_layer = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )
        self.output_layer = LinearStack(
            in_features=features_dim[1],
            out_features=features_dim[2],
            activation_fn=activation_fn,
            hidden_features=features_dim[1],
            dropout=self.dropout
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.env_features_len,
            num_assets=self.num_assets,
            include_weekends=self.include_weekends
        )
        variables = self.split_variables(observations)

        selected_variables = self.vsn(variables)
        recurrent_output, _ = self.recurrent_layer(selected_variables)
        recurrent_output = recurrent_output[:, -1, :]
        output = self.output_layer(recurrent_output)

        return output

    @classmethod
    def split_variables(cls, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        variables = dict()
        for key, value in observations.items():
            if key == 'env_features':
                variables['private_features'] = value
            elif key == '1d':
                num_assets = value.shape[3]
                for asset_idx in range(num_assets):
                    asset_values = value[..., asset_idx, :]
                    batch_size, window_size = asset_values.shape[:2]
                    asset_values = asset_values.view(batch_size, window_size, -1)
                    variables[f'public_features_{asset_idx}'] = asset_values
            else:
                raise RuntimeError(f'Unsupported observation type: {key}')

        return variables


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
            rnn_layer_type: nn.Module
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
            activation_fn()
        )
        self.public_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )
        self.public_attention = nn.MultiheadAttention(
            embed_dim=features_dim[1],
            num_heads=8
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
        self.private_attention = nn.MultiheadAttention(
            embed_dim=features_dim[1],
            num_heads=8
        )

        self.output_attention = nn.MultiheadAttention(
            embed_dim=features_dim[1],
            num_heads=8
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(features_dim[1], features_dim[-1]),
            activation_fn()
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
        # public_input = public_input[-1, :, :]

        private_input = self.private_mlp(private_input)
        private_input, _ = self.private_recurrent(private_input)
        private_input = private_input.transpose(0, 1)
        private_input, _ = self.private_attention(private_input, private_input, private_input)
        # private_input = private_input[-1, :, :]

        output = public_input + private_input
        output, _ = self.output_attention(output, output, output)
        # output = torch.cat([public_input, private_input], dim=-1)
        output = output[-1, :, :]
        # output = output.reshape(batch_size, -1)
        output = self.output_mlp(output)

        return output
