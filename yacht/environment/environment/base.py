from data.loaders import BaseDataLoader
from environment.portfolios import BasePortfolio


class BaseEnvironment:
    def __init__(
            self,
            portfolio: BasePortfolio,
            reward_scheme,
            action_scheme,
            data_loader: BaseDataLoader
    ):
        self.portfolio = portfolio
        self.reward_scheme = reward_scheme
        self.action_scheme = action_scheme
        self.data_loader = data_loader

    def next_batch(self) -> dict:
        return self.data_loader.next_batch()
