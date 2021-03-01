from .networks import Network
from environment.environment import Environment
from .networks.base import BaseNetwork


class BaseAgent:
    def __init__(
            self,
            environment: Environment,
            window_size: int,
            layers_config,
            steps: int,
            device: str,
    ):
        self.environment = environment
        self.window_size = window_size
        self.layers_config = layers_config
        self.steps = steps
        self.device = device

        self.network = self.build_network()

    def train(self):
        raise NotImplementedError()

    def build_network(self) -> BaseNetwork:
        raise NotImplementedError()
