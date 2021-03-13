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



