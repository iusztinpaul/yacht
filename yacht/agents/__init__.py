from .base import *
# from .cnn import CNNAgent

from config import Config
from environment.environment import Environment


def build_agent(environment: Environment, config: Config) -> BaseAgent:
    # agent = CNNAgent(
    #     environment=environment,
    #     window_size=config.input_config.window_size,
    #     layers_config=dict(),
    #     steps=config.training_config.steps,
    #     device='gpu',
    # )

    return None
