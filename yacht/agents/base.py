from .networks import Network
from environment import BaseEnvironment


class BaseAgent:
    def __init__(
            self,
            environment: BaseEnvironment,
            window_size: int,
            layers_config,
            steps: int,
            device: str,
    ):
        self.environment = environment
        self.steps = steps
        self.network = Network(
            feature_num=environment.features_num,
            assets_num=environment.assets_num,
            window_size=window_size,
            layers_config=layers_config,
            device=device
        )

    def train(self):
        raise NotImplementedError()
