from .base import *

from config import Config
from environment.environment import Environment
from .eiie import EIIEAgent

agents = {
    'EIIEAgent': EIIEAgent
}


def build_agent(environment: Environment, config: Config) -> BaseAgent:
    agent_class = agents[config.training_config.agent]

    agent = agent_class(
        environment=environment,
        config=config
    )

    return agent
