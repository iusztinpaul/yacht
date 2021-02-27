from .base import *
from agents.networks import Network
from config import Config
from environment import BaseEnvironment


class Agent(BaseAgent):
    pass


def build_agent(environment: BaseEnvironment, config: Config) -> BaseAgent:
    agent = Agent(
        environment=environment,
        window_size=config.input_config.window_size,
        layers_config=dict(),
        steps=config.training_config.steps,
        device='gpu',
    )

    return agent
