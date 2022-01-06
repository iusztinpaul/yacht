from typing import Tuple, Dict, Optional

import torch

from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class SupervisedMlpExtractor(MlpExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        activation_fn = kwargs['activation_fn']
        device = kwargs['device']
        
        self.latent_dim_supervised = self.latent_dim_pi
        supervised_net = []
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                supervised_net.append(
                    nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
                )
                supervised_net.append(activation_fn())
        self.supervised_net = nn.Sequential(*supervised_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent), self.supervised_net(shared_latent)


class SimplifiedVariableSelectionNetwork(nn.Module):
    def __init__(
            self,
            public_features_len: int,
            private_features_len: Optional[int],
            num_assets: int,
            hidden_size: int,
            activation_fn: nn.Module,
            dropout: Optional[float] = None,
    ):
        super().__init__()

        self.public_features_len = public_features_len
        self.private_features_len = private_features_len
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.input_sizes = {
            f'public_features_{asset_idx}': self.public_features_len for asset_idx in range(self.num_assets)
        }
        if self.private_features_len is not None:
            self.input_sizes['private_features'] = self.private_features_len

        self.single_variable_layers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            self.single_variable_layers[name] = LinearStack(
                in_features=input_size,
                hidden_features=self.hidden_size,
                out_features=self.hidden_size,
                activation_fn=activation_fn,
                dropout=self.dropout
            )

        self.flattened_variables_layer = LinearStack(
            in_features=self.input_size_total,
            hidden_features=hidden_size,
            out_features=self.num_inputs,
            activation_fn=activation_fn,
            dropout=self.dropout,
            n=2
        )
        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size for size in self.input_sizes.values())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor]):
        var_outputs = []
        weight_inputs = []
        for name in self.input_sizes.keys():
            variable_embedding = x[name]
            weight_inputs.append(variable_embedding)
            var_outputs.append(self.single_variable_layers[name](variable_embedding))
        var_outputs = torch.stack(var_outputs, dim=-1)

        # calculate variable weights
        flat_embedding = torch.cat(weight_inputs, dim=-1)
        sparse_weights = self.flattened_variables_layer(flat_embedding)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(dim=-1)

        return outputs


class LinearStack(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation_fn: nn.Module,
            n: int = 1,
            hidden_features: Optional[int] = None,
            dropout: Optional[float] = None
    ):
        super().__init__()

        if hidden_features is None or n == 1:
            hidden_features = out_features
        modules = []
        for i in range(n):
            if i == 0:
                modules.append(nn.Linear(in_features=in_features, out_features=hidden_features))
            elif 1 < i < n - 1:
                modules.append(nn.Linear(in_features=hidden_features, out_features=hidden_features))
            else:
                modules.append(nn.Linear(in_features=hidden_features, out_features=out_features))
            modules.append(activation_fn())
            if dropout is not None and dropout > 0:
                modules.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*modules)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "weight" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
