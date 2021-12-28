from copy import deepcopy
from typing import Tuple

import torch as th

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

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent), self.supervised_net(shared_latent)
