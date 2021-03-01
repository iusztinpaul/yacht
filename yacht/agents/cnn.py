from agents import BaseAgent, BaseNetwork
from agents.networks import CNNNetwork


class CNNAgent(BaseAgent):
    def build_network(self) -> BaseNetwork:
        return CNNNetwork(
            feature_num=self.environment.features_num,
            assets_num=self.environment.assets_num,
            window_size=self.window_size,
            layers_config=self.layers_config,
            device=self.device
        )
