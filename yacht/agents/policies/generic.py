from typing import Callable, Type, Optional, Dict, Any

import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GenericActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            activation_fn: Type[nn.Module],
            backbone_class: Optional[nn.Module],
            backbone_kwargs: Optional[Dict[str, Any]],
            features_extractor_class: Optional[Type[BaseFeaturesExtractor]],
            features_extractor_kwargs: Optional[Dict[str, Any]],
            **kwargs,
    ):
        self.backbone_class = backbone_class
        self.backbone_kwargs = backbone_kwargs

        super(GenericActorCriticPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=None,
            activation_fn=activation_fn,
            normalize_images=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )
        # TODO: What is orthogonal initialization ? Why should it be disabled ?
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self.backbone_class(
            activation_fn=self.activation_fn,
            **self.backbone_kwargs
        )
