import logging

import torch

from agents import BaseAgent
from agents.strategies import EIIENetwork
from config import Config
from environment.environment import Environment


logger = logging.getLogger(__file__)


class EIIEAgent(BaseAgent):
    def build_strategy(self, environment: Environment, config: Config):
        market = environment.market

        network = EIIENetwork(
            num_features=len(market.features),
            num_assets=len(market.tickers),
            window_size=config.input_config.window_size,
            commission=market.commission
        )
        network = network.to(config.hardware_config.device)

        return network

    def train(self):
        training_config = self.config.training_config

        logger.info('Training...')
        for step in range(self.start_training_step, training_config.steps):
            self.optimizer.zero_grad()

            X, y, last_w, batch_new_w_datetime = self.get_train_data()

            new_w = self.strategy(X, last_w)
            loss = self.strategy.compute_loss(new_w, y)

            loss.backward()
            self.optimizer.step()

            if step % training_config.learning_rate_decay_steps:
                self.scheduler.step()

            new_w = new_w.detach().cpu().numpy()
            self.environment.set_portfolio_weights(batch_new_w_datetime, new_w)

            if step % training_config.validation_every_step == 0 and step != 0:
                self.strategy.train(mode=False)
                with torch.no_grad():
                    X_val, y_val, last_w_val, batch_new_w_datetime_val = self.get_validation_data()

                    new_w_val = self.strategy(X_val, last_w_val)
                    metrics = self.strategy.compute_metrics(new_w_val, y_val)
                    self.log_metrics(step, metrics)

                self.strategy.train(mode=True)

            if step % training_config.save_every_step == 0 and step != 0:
                self.save_network(step, loss)

    def log_metrics(self, step: int, metrics: dict):
        self.log(step, f'Portfolio value: {metrics["portfolio_value"].detach().cpu().numpy()}')
        self.log(step, f'Sharp ratio: {metrics["sharp_ratio"].detach().cpu().numpy()}\n')
