from abc import ABC

from gym import Space, spaces
import numpy as np

from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig


class ActionSchema(ABC):
    def __init__(self, num_assets: int, max_units_per_asset: int):
        self.num_assets = num_assets
        self.max_units_per_asset = max_units_per_asset

    def get_action_space(self) -> Space:
        raise NotImplementedError()

    def get_value(self, action: np.array) -> np.array:
        raise NotImplementedError()


class DiscreteActionScheme(ActionSchema):
    pass


class ContinuousActionScheme(ActionSchema):
    def get_action_space(self) -> Space:
        return spaces.Box(low=-1, high=1, shape=(self.num_assets, ))

    def get_value(self, action: np.array) -> np.array:
        return (action * self.max_units_per_asset).astype(np.int32)


#######################################################################################################################


action_schema_registry = {
    'DiscreteActionScheme': DiscreteActionScheme,
    'ContinuousActionScheme': ContinuousActionScheme
}


def build_action_schema(config: Config):
    env_config: EnvironmentConfig = config.environment
    action_schema_class = action_schema_registry[env_config.action_schema]

    if action_schema_class == ContinuousActionScheme:
        assert env_config.max_units_per_asset > 0

        # TODO: Support multiple assets for action schema.
        return action_schema_class(
            num_assets=1,
            max_units_per_asset=env_config.max_units_per_asset
        )
    else:
        raise NotImplementedError()
