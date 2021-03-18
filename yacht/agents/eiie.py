import logging

import torch
from torch import optim

from agents import BaseAgent
from agents.networks import EIIENetwork
from config import Config
from environment.environment import Environment


logger = logging.getLogger(__file__)


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

    def train(self):
        training_config = self.config.training_config
        hardware_config = self.config.hardware_config

        optimizer = optim.Adam(
            params=self.network.params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=training_config.learning_rate_decay,
        )

        for step in range(training_config.steps):
            self.network.train(mode=True)
            optimizer.zero_grad()

            X, y, last_w, batch_new_w_datetime = self.environment.next_batch_train()

            X = torch.from_numpy(X).to(hardware_config.device)
            y = torch.from_numpy(y).to(hardware_config.device)
            last_w = torch.from_numpy(last_w).to(hardware_config.device)

            new_w = self.network(X, last_w)
            loss = self.network.compute_loss(new_w, y)

            loss.backward()
            optimizer.step()

            if step % training_config.learning_rate_decay_steps:
                scheduler.step()

            new_w = new_w.detach().cpu().numpy()
            self.environment.set_portfolio_weights(batch_new_w_datetime, new_w)

            if step % training_config.log_steps == 0:
                self.network.train(mode=False)
                with torch.no_grad():
                    X_val, y_val, last_w_val, batch_new_w_datetime_val = self.environment.next_batch_val()

                    X_val = torch.from_numpy(X_val).to(hardware_config.device)
                    y_val = torch.from_numpy(y_val).to(hardware_config.device)
                    last_w_val = torch.from_numpy(last_w_val).to(hardware_config.device)

                    new_w_val = self.network(X_val, last_w_val)
                    metrics = self.network.compute_metrics(new_w_val, y_val)
                    self.log_metrics(step, metrics)

                    self.network.compute_metrics(step)

    def log_metrics(self, step: int, metrics: dict):
        logging.info(f' Step [{step}] - Portfolio value: {metrics["portfolio_value"].detach().cpu().numpy()}')
        logging.info(f' Step [{step}] - Sharp ratio: {metrics["sharp_ratio"].detach().cpu().numpy()}\n')
