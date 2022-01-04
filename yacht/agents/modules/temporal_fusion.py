from typing import Dict, List, Tuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from yacht.agents.misc import unflatten_observations


class DayTemporalFusionFeatureExtractor(BaseFeaturesExtractor):
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
            dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim[-1])

        assert len(features_dim) >= 3
        assert len(set(features_dim[1:-1])) == 1, 'The features_dim of the recurrent layers should be equal.'

        self.window_size = window_size
        self.intervals = intervals
        self.public_features = features
        self.private_features_len = env_features_len
        self.num_assets = num_assets
        self.include_weekends = include_weekends
        self.num_rnn_layers = len(features_dim[1:-1])
        self.dropout = dropout if dropout else None

        # Step 1: Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            public_features_len=len(self.public_features),
            private_features_len=self.private_features_len,
            num_assets=self.num_assets,
            hidden_size=features_dim[0],
            dropout=self.dropout
        )
        # Step 2: Recurrent layers.
        self.recurrent_layer = rnn_layer_type(
            features_dim[0],
            features_dim[1],
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=self.dropout if self.num_rnn_layers > 1 else 0
        )
        self.post_recurrent_gate_add_norm = GateAddNorm(
            input_size=features_dim[1],
            hidden_size=features_dim[1],
            trainable_add=False,
            dropout=self.dropout
        )
        # Step 3: Attention layers.
        self.pre_attn_grn = GatedResidualNetwork(
            input_size=features_dim[1],
            hidden_size=features_dim[1],
            output_size=features_dim[1],
            dropout=self.dropout
        )
        self.multihead_attn = InterpretableMultiHeadAttention(
            n_head=1,
            d_model=features_dim[1],
            dropout=self.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            input_size=features_dim[1],
            hidden_size=features_dim[1],
            dropout=self.dropout,
            trainable_add=False
        )
        # Step 4: Final output layers.
        self.output_grn = GatedResidualNetwork(
            input_size=features_dim[1],
            hidden_size=features_dim[1],
            output_size=features_dim[1],
            dropout=self.dropout
        )
        self.out_gate_add_norm = GateAddNorm(
            input_size=features_dim[1],
            hidden_size=features_dim[1],
            trainable_add=False,
            dropout=None
        )
        self.output_layer = nn.Linear(features_dim[1], features_dim[2])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = unflatten_observations(
            observations=observations,
            intervals=self.intervals,
            num_env_features=self.private_features_len,
            num_assets=self.num_assets,
            include_weekends=self.include_weekends
        )
        # Step 1: Variable Selection Network.
        variables = self.split_variables(observations)
        selected_variables = self.vsn(variables)
        # Step 2: Recurrent layers.
        recurrent_output, _ = self.recurrent_layer(selected_variables)
        # TODO : Why they used the gate & add norm layers splitted ?
        recurrent_output = self.post_recurrent_gate_add_norm(recurrent_output, selected_variables)
        # Step 3: Attention layers.
        attention_input = self.pre_attn_grn(recurrent_output)
        attention_output, _ = self.multihead_attn(
            q=attention_input[:, -1:],
            k=attention_input,
            v=attention_input
        )
        attention_output = self.post_attn_gate_norm(attention_output, attention_input[:, -1:])
        # Step 4: Final output layers.
        output = self.output_grn(attention_output)
        output = self.out_gate_add_norm(output, recurrent_output[:, -1:])
        output = self.output_layer(output)
        output = torch.squeeze(output, dim=1)

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


class VariableSelectionNetwork(nn.Module):
    def __init__(
            self,
            public_features_len: int,
            private_features_len: int,
            num_assets: int,
            hidden_size: int,
            dropout: float = 0.1,
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.public_features_len = public_features_len
        self.private_features_len = private_features_len
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.input_sizes = {
            f'public_features_{asset_idx}': self.public_features_len for asset_idx in range(self.num_assets)
        }
        self.input_sizes['private_features'] = self.private_features_len

        self.single_variable_grns = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            self.single_variable_grns[name] = GatedResidualNetwork(
                input_size=input_size,
                hidden_size=min(input_size, self.hidden_size),
                output_size=self.hidden_size,
                dropout=self.dropout,
            )

        self.flattened_grn = GatedResidualNetwork(
            input_size=self.input_size_total,
            hidden_size=min(self.hidden_size, self.num_inputs),
            output_size=self.num_inputs,
            dropout=self.dropout,
            residual=False,
        )
        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size for size in self.input_sizes.values())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        var_outputs = []
        weight_inputs = []
        for name in self.input_sizes.keys():
            variable_embedding = x[name]
            weight_inputs.append(variable_embedding)
            var_outputs.append(self.single_variable_grns[name](variable_embedding))
        var_outputs = torch.stack(var_outputs, dim=-1)

        # calculate variable weights
        flat_embedding = torch.cat(weight_inputs, dim=-1)
        sparse_weights = self.flattened_grn(flat_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(dim=-1)

        return outputs


class GatedResidualNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
            context_size: int = None,
            residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)

        return x


class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)

        return output


class GateAddNorm(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = None,
            skip_size: int = None,
            trainable_add: bool = False,
            dropout: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)

        return output


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)

        return x


class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0

        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            if self.dropout is not None:
                head = self.dropout(head)
            heads.append(head)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
