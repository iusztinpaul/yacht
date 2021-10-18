from typing import List, Optional

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from yacht.agents.misc import unflatten_observations


class SlimTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            features_in: int,
            features_out: int,
            nhead: int,
            activation_fn: nn.Module,
            dropout=0.1
    ):
        super(SlimTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(features_in, nhead, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        # self.attn_norm = LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear = nn.Linear(features_in, features_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_fn()

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.attn_dropout(src2)
        # src = self.attn_norm(src)
        src = self.dropout(self.activation(self.linear(src)))

        return src


class TransformerFeatureExtractor(BaseFeaturesExtractor):
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
            drop_out_p: float = 0.5
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) == 3

        self.window_size = window_size
        self.intervals = intervals
        self.features = features
        self.env_features_len = env_features_len
        self.num_assets = num_assets

        self.public_transformer = nn.Sequential(
            SlimTransformerEncoderLayer(
                features_in=len(features),
                features_out=features_dim[0],
                nhead=1,
                activation_fn=activation_fn,
                dropout=drop_out_p
            ),
            SlimTransformerEncoderLayer(
                features_in=features_dim[0],
                features_out=features_dim[1],
                nhead=8,
                activation_fn=activation_fn,
                dropout=drop_out_p
            ),
            SlimTransformerEncoderLayer(
                features_in=features_dim[1],
                features_out=features_dim[1],
                nhead=8,
                activation_fn=activation_fn,
                dropout=drop_out_p
            )
        )
        self.private_transformer = nn.Sequential(
            SlimTransformerEncoderLayer(
                features_in=env_features_len,
                features_out=features_dim[1],
                nhead=1,
                activation_fn=activation_fn,
                dropout=drop_out_p
            ),
            SlimTransformerEncoderLayer(
                features_in=features_dim[1],
                features_out=features_dim[1],
                nhead=1,
                activation_fn=activation_fn,
                dropout=drop_out_p
            )
        )
        # self.shared_transformer = nn.Sequential(
        #     SlimTransformerEncoderLayer(
        #         features_in=features_dim[1],
        #         features_out=features_dim[2],
        #         nhead=8,
        #         activation_fn=activation_fn,
        #         dropout=drop_out_p
        #     )
        # )
        self.shared_mlp = nn.Sequential(
            nn.Linear(
                in_features=2*features_dim[1],
                out_features=features_dim[2]
            ),
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
        public_input = public_input.transpose(0, 1)

        batch_size, window_size, env_features = observations['env_features'].shape
        private_input = observations['env_features']
        private_input = private_input.transpose(0, 1)

        public_input = self.public_transformer(public_input)
        public_input = public_input[-1, :, :]
        private_input = self.private_transformer(private_input)
        private_input = private_input[-1, :, :]

        shared = torch.cat([public_input, private_input], dim=-1)
        # shared_input = self.shared_transformer(shared_input)
        shared = self.shared_mlp(shared)

        return shared
