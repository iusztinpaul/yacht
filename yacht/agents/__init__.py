from .base import *

from config import Config
from environment.environment import Environment
from .eiie import TrainEIIEAgent, BackTestEIIEAgent

train_agents = {
    'EIIEAgent': TrainEIIEAgent
}
back_test_agents = {
    'EIIEAgent': BackTestEIIEAgent
}


def build_train_agent(
        environment: Environment,
        config: Config,
        storage_path: str,
        resume_training: bool
) -> TrainBaseAgent:
    agent_class = train_agents[config.training_config.agent]

    agent = agent_class(
        environment=environment,
        config=config,
        storage_path=storage_path,
        resume_training=resume_training
    )

    return agent


def build_back_test_agent(
        environment: Environment,
        config: Config,
        storage_path: str,
) -> BackTestBaseAgent:
    agent_class = back_test_agents[config.training_config.agent]

    agent = agent_class(
        environment=environment,
        config=config,
        storage_path=storage_path,
    )

    return agent
