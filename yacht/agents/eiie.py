from torch import optim

from agents import BaseAgent
from agents.networks import EIIENetwork
from config import Config, TrainingConfig
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

    def train(self, training_config: TrainingConfig):
        optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        # TODO: What to add at last_epoch ???
        optimizer = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=training_config.learning_rate_decay,
        )

        for step in range(training_config.steps):
            optimizer.zero_grad()

            X, y, last_w = self.environment.next_batch()
            loss = self.network(X, y, last_w)

            loss.backward()
            optimizer.step()
