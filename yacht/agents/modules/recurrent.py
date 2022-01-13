from typing import List, Optional, Dict

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from yacht.agents.misc import unflatten_observations
from yacht.agents.modules.torch_layers import SimplifiedVariableSelectionNetwork, LinearStack, Resample, AddNorm


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
            dropout: Optional[float] = None,
            attention_head_size: int = 1,
            add_attention: bool = False,
            add_normalization: bool = False,
            add_output_vsn: bool = False,
            add_residual: bool = False
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
        self.dropout = dropout if dropout and dropout > 0 else None
        self.attention_head_size = attention_head_size
        self.add_attention = add_attention
        self.add_normalization = add_normalization
        self.add_output_vsn = add_output_vsn
        self.add_residual = add_residual

        self.public_vsn = SimplifiedVariableSelectionNetwork(
            public_features_len=len(self.features),
            private_features_len=None,
            num_assets=self.num_assets,
            hidden_features=features_dim[0],
            activation_fn=activation_fn,
            dropout=self.dropout,
            layers_type='linear',
            add_normalization=self.add_normalization,
            add_residual=self.add_residual
        )
        self.public_recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )
        if self.add_residual is True:
            self.public_resample = Resample(
                in_features=features_dim[0],
                out_features=features_dim[1],
                activation_fn=activation_fn,
                trainable_add=False,
                dropout=self.dropout
            )
            self.public_add_norm = AddNorm(
                out_features=features_dim[1],
                add_normalization=self.add_normalization,
                add_residual=self.add_residual
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
        if self.add_residual is True:
            self.private_resample = Resample(
                in_features=features_dim[0],
                out_features=features_dim[1],
                activation_fn=activation_fn,
                trainable_add=False,
                dropout=self.dropout
            )
            self.private_add_norm = AddNorm(
                out_features=features_dim[1],
                add_normalization=self.add_normalization,
                add_residual=self.add_residual
            )

        if self.add_output_vsn is True:
            self.output_vsn = SimplifiedVariableSelectionNetwork(
                public_features_len=features_dim[1],
                private_features_len=features_dim[1],
                num_assets=1,
                hidden_features=features_dim[1],
                activation_fn=activation_fn,
                dropout=self.dropout,
                layers_type='linear',
                add_normalization=self.add_normalization,
                add_residual=self.add_residual
            )
        if self.add_attention is True:
            self.output_attn = nn.MultiheadAttention(
                embed_dim=features_dim[1] if self.add_output_vsn is True else features_dim[1] * 2,
                num_heads=self.attention_head_size,
                dropout=dropout
            )
        if self.add_residual is True:
            self.output_add_norm = AddNorm(
                out_features=features_dim[1] if self.add_output_vsn is True else features_dim[1] * 2,
                add_normalization=self.add_normalization,
                add_residual=self.add_residual
            )
        self.output_mlp = LinearStack(
            in_features=features_dim[1] if self.add_output_vsn is True else features_dim[1] * 2,
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
        public_vsn_output = self.public_vsn(public_variables)
        public_rnn_output, _ = self.public_recurrent(public_vsn_output)
        if self.add_residual is True:
            residual = self.public_resample(public_vsn_output)
            public_rnn_output = self.public_add_norm(public_rnn_output, residual)

        private_mlp_output = self.private_mlp(private_input)
        private_rnn_output, _ = self.private_recurrent(private_mlp_output)
        if self.add_residual is True:
            residual = self.private_resample(private_mlp_output)
            private_rnn_output = self.private_add_norm(private_rnn_output, residual)

        if self.add_output_vsn:
            variables = {
                'public_features_0': public_rnn_output,
                'private_features': private_rnn_output
            }
            aggregated_output = self.output_vsn(variables)
        else:
            aggregated_output = torch.cat([public_rnn_output, private_rnn_output], dim=-1)
        if self.add_attention is True:
            output, _ = self.output_attn(
                query=aggregated_output[:, -1:],
                key=aggregated_output,
                value=aggregated_output
            )
            if self.add_residual is True:
                output = self.output_add_norm(output, aggregated_output[:, -1:])
        else:
            output = aggregated_output[:, -1, :]
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
            dropout: Optional[float] = None,
            attention_head_size: int = 1,
            add_attention: bool = False,
            add_normalization: bool = False,
            add_residual: bool = False
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
        self.dropout = dropout if dropout and dropout > 0 else None
        self.attention_head_size = attention_head_size
        self.add_attention = add_attention
        self.add_normalization = add_normalization
        self.add_residual = add_residual

        self.vsn = SimplifiedVariableSelectionNetwork(
            public_features_len=len(self.features),
            private_features_len=None,
            num_assets=self.num_assets,
            hidden_features=features_dim[0],
            activation_fn=activation_fn,
            dropout=self.dropout,
            layers_type='linear',
            add_normalization=self.add_normalization,
            add_residual=self.add_residual
        )
        self.recurrent = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True
        )
        if self.add_residual is True:
            self.resample = Resample(
                in_features=features_dim[0],
                out_features=features_dim[1],
                activation_fn=activation_fn,
                trainable_add=False,
                dropout=self.dropout
            )
            self.add_norm = AddNorm(
                out_features=features_dim[1],
                add_normalization=self.add_normalization,
                add_residual=self.add_residual
            )

        if self.add_attention is True:
            self.output_attn = nn.MultiheadAttention(
                embed_dim=features_dim[1],
                num_heads=self.attention_head_size,
                dropout=dropout
            )
        if self.add_residual is True:
            self.output_add_norm = AddNorm(
                out_features=features_dim[1],
                add_normalization=self.add_normalization,
                add_residual=self.add_residual
            )
        self.output_mlp = LinearStack(
            in_features=features_dim[1],
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
        batch_size, _, _ = observations['env_features'].shape
        variables = self.split_variables(observations)

        vsn_output = self.vsn(variables)
        rnn_output, _ = self.recurrent(vsn_output)
        if self.add_residual is True:
            residual = self.resample(vsn_output)
            rnn_output = self.add_norm(rnn_output, residual)
        if self.add_attention is True:
            output, _ = self.output_attn(
                query=rnn_output[:, -1:],
                key=rnn_output,
                value=rnn_output
            )
            if self.add_residual is True:
                output = self.output_add_norm(output, rnn_output[:, -1:])
        else:
            output = rnn_output[:, -1, :]
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
