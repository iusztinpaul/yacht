# [
#     {
#         'activation_function': 'relu', 'filter_number': 3, 'filter_shape': [1, 2], 'padding': 'valid',
#         'regularizer': None,
#         'strides': [1, 1], 'type': 'ConvLayer', 'weight_decay': 0.0
#     },
#     {
#         'activation_function': 'relu', 'filter_number': 10, 'regularizer': 'L2', 'type': 'EIIE_Dense',
#         'weight_decay': 5e-09
#     },
#     {
#         'regularizer': 'L2', 'type': 'EIIE_Output_WithW', 'weight_decay': 5e-08
#     }
# ]
import torch

from torch import nn

from agents.networks import layers


class EIIENetwork(nn.Module):
    def __init__(
            self,
            num_features: int,
            num_assets: int,
            window_size: int,
            name='EIIENetwork'
    ):
        super().__init__()
        self.name = name

        self.eiie_layer = layers.EIIECNN(
            num_features=num_features,
            num_assets=num_assets,
            window_size=window_size
        )

    def forward(self, X, y, previous_w):
        new_w = self.eiie_layer(X, previous_w)

        if self.training:
            return self.loss(new_w, y)
        else:
            return new_w

    def loss(self, new_w, y):
        batch_num = y.shape[0]

        future_price = torch.cat(
            [
                torch.ones(size=(batch_num, 1)).to(y.device),
                y[:, 0, :]
            ],
            dim=1
        )

        total_assets_value = torch.sum(future_price * new_w, dim=1).reshape(batch_num, 1)
        future_omega = future_price * new_w / total_assets_value
