from stable_baselines3.common.policies import ActorCriticPolicy as StableB3ActorCriticPolicy
from torch import nn


class ActorCriticPolicy(StableB3ActorCriticPolicy):
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.GRU):
            nn.init.orthogonal_(module.weight_hh_l0, gain=gain)
            nn.init.orthogonal_(module.weight_ih_l0, gain=gain)
            if module.bias is not None:
                module.bias_hh_l0.data.fill_(0.0)
                module.bias_ih_l0.data.fill_(0.0)
        elif isinstance(module, (nn.RNN, nn.LSTM)):
            raise RuntimeError('"init_weights()" not implemented for (RNN, LSTM).')
