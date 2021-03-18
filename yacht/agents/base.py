from config import Config, TrainingConfig
from environment.environment import Environment


class BaseAgent:
    def __init__(self, environment: Environment, config: Config):
        self.environment = environment
        self.config = config

        self.network = self.build_network(environment, config)

    def build_network(self, environment: Environment, config: Config):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
