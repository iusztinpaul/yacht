from abc import ABCMeta

from environment.environment.base import BaseEnvironment


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment
