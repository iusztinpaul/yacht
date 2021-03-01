from .base import BaseNetwork

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


class CNNNetwork(BaseNetwork):
    pass
    # def __init__(self):
    #     self.eiie_cnn = EIIECNN(
    #         feature_num=feature_num,
    #         assets_num=assets_num,
    #         window_size=window_size,
    #         layers_config=layers_config,
    #         device=device
    #     )
