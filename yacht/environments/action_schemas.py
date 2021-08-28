from abc import ABC
from typing import List

from gym import Space, spaces
import numpy as np

from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import ChooseAssetDataset


class ActionSchema(ABC):
    def __init__(self, num_assets: int):
        self.num_assets = num_assets

    def get_action_space(self) -> Space:
        raise NotImplementedError()

    def get_value(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class DiscreteActionScheme(ActionSchema):
    def __init__(self, num_assets: int, possibilities: List[float]):
        super().__init__(num_assets=num_assets)

        self.possibilities = np.array(possibilities)

    def get_action_space(self) -> Space:
        return spaces.MultiDiscrete(nvec=(self.num_assets, len(self.possibilities)))

    def get_value(self, action: np.ndarray) -> np.ndarray:
        return self.possibilities[action]


class ContinuousFloatActionSchema(ActionSchema):
    def __init__(self, num_assets: int, action_scaling_factor: int):
        super().__init__(num_assets=num_assets)

        self.action_scaling_factor = action_scaling_factor

    def get_action_space(self) -> Space:
        return spaces.Box(low=-1, high=1, shape=(self.num_assets, ))

    def get_value(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_scaling_factor


class ContinuousIntegerActionSchema(ContinuousFloatActionSchema):
    def get_value(self, action: np.ndarray) -> np.ndarray:
        action = super().get_value(action)

        return action.astype(np.int32)


#######################################################################################################################


action_schema_registry = {
    'DiscreteActionScheme': DiscreteActionScheme,
    'ContinuousFloatActionSchema': ContinuousFloatActionSchema,
    'ContinuousIntegerActionSchema': ContinuousIntegerActionSchema
}


def build_action_schema(config: Config, dataset: ChooseAssetDataset):
    env_config: EnvironmentConfig = config.environment
    action_schema_class = action_schema_registry[env_config.action_schema]

    if action_schema_class in (ContinuousIntegerActionSchema, ContinuousFloatActionSchema):
        assert env_config.action_scaling_factor > 0

        return action_schema_class(
            num_assets=dataset.num_assets,
            action_scaling_factor=env_config.action_scaling_factor
        )
    else:
        assert len(env_config.possibilities) > 0

        return action_schema_class(
            num_assets=dataset.num_assets,
            possibilities=list(env_config.possibilities)
        )
