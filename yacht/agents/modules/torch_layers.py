from typing import Tuple, Dict, Optional

import torch

from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn as nn
from torch.nn import functional as F


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
            hidden_features: int,
            activation_fn: nn.Module,
            dropout: Optional[float] = None,
            layers_type: str = 'linear',
            add_normalization: bool = False,
            add_residual: bool = False
    ):
        super().__init__()

        self.public_features_len = public_features_len
        self.private_features_len = private_features_len
        self.num_assets = num_assets
        self.hidden_features = hidden_features
        self.dropout = dropout
        self.add_normalization = add_normalization
        self.add_residual = add_residual

        self.input_sizes = {
            f'public_features_{asset_idx}': self.public_features_len for asset_idx in range(self.num_assets)
        }
        if self.private_features_len is not None:
            self.input_sizes['private_features'] = self.private_features_len

        self.single_variable_layers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if layers_type == 'linear':
                self.single_variable_layers[name] = LinearStack(
                    in_features=input_size,
                    out_features=self.hidden_features,
                    activation_fn=activation_fn,
                    dropout=self.dropout,
                    n=2
                )
            elif layers_type == 'grn':
                self.single_variable_layers[name] = SimplifiedGatedResidualNetwork(
                    in_features=input_size,
                    out_features=self.hidden_features,
                    activation_fn=activation_fn,
                    dropout=self.dropout,
                    add_normalization=self.add_normalization,
                    add_residual=self.add_residual
                )
            else:
                raise RuntimeError(f'Wrong layers_type={layers_type}')

        if layers_type == 'linear':
            self.flattened_variables_layer = LinearStack(
                in_features=self.input_size_total,
                hidden_features=self.hidden_features,
                out_features=self.num_inputs,
                activation_fn=activation_fn,
                dropout=self.dropout,
                n=2
            )
        elif layers_type == 'grn':
            self.flattened_variables_layer = SimplifiedGatedResidualNetwork(
                in_features=self.input_size_total,
                out_features=self.num_inputs,
                activation_fn=activation_fn,
                dropout=self.dropout,
                add_normalization=self.add_normalization
            )
        else:
            raise RuntimeError(f'Wrong layers_type={layers_type}')
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
            activation_fn: Optional[nn.Module] = None,
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
            if activation_fn is not None:
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


class SimplifiedGatedResidualNetwork(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation_fn: nn.Module,
            dropout: float = 0.1,
            add_normalization: bool = False,
            add_residual: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.add_normalization = add_normalization
        self.add_residual = add_residual

        if self.in_features != self.out_features:
            self.resample = Resample(
                in_features=self.in_features,
                out_features=self.out_features,
                activation_fn=activation_fn,
                trainable_add=True,
                dropout=self.dropout
            )
        self.lin1 = LinearStack(
            in_features=self.in_features,
            out_features=self.out_features,
            activation_fn=activation_fn,
            dropout=self.dropout
        )
        self.lin2 = LinearStack(
            in_features=self.out_features,
            out_features=self.out_features,
            activation_fn=None,
            dropout=self.dropout
        )
        self.glu = GatedLinearUnit(
            in_features=self.out_features,
            out_features=self.out_features
        )
        if self.add_normalization is True:
            self.norm = nn.LayerNorm(self.out_features)

    def forward(self, x):
        if self.add_residual is True and self.in_features != self.out_features:
            residual = self.resample(x)
        else:
            residual = x

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.glu(x)
        if self.add_residual is True:
            x = x + residual
        if self.add_normalization is True:
            x = self.norm(x)

        return x


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, in_features: int, out_features: int = None):
        super().__init__()

        self.out_features = out_features or in_features
        self.fc = LinearStack(
            in_features=in_features,
            out_features=self.out_features * 2,
            activation_fn=None
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.glu(x, dim=-1)

        return x


class Resample(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation_fn: nn.Module,
            trainable_add: bool = True,
            dropout: Optional[float] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.trainable_add = trainable_add
        self.out_features = out_features
        self.dropout = dropout

        if self.in_features != self.out_features:
            self.resample = LinearStack(
                in_features=self.in_features,
                out_features=self.out_features,
                activation_fn=activation_fn,
                dropout=self.dropout
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float))
            self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_features != self.out_features:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        return x


class AddNorm(nn.Module):
    def __init__(
            self,
            out_features: int,
            add_normalization: bool,
            add_residual: bool
    ):
        super().__init__()
        
        self.out_features = out_features
        self.add_normalization = add_normalization
        self.add_residual = add_residual
        
        if self.add_normalization is True:
            self.norm_layer = nn.LayerNorm(self.out_features)
            
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if self.add_residual:
            x = x + residual
        if self.add_normalization:
            x = self.norm_layer(x)

        return x
