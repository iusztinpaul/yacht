from agents import BaseAgent
from agents.networks import EIIENetwork
from config import Config
from environment.environment import Environment


class EIIEAgent(BaseAgent):
    def build_network(self, environment: Environment, config: Config):
        market = environment.market

        network = EIIENetwork(
            num_features=len(market.features),
            num_assets=len(market.tickers),
            window_size=config.input_config.window_size,
            commission=market.commission
        )
        network = network.to(config.hardware_config.device)

        return network
