from abc import ABC, abstractmethod
from typing import List, Optional

from gym import Space, spaces
import numpy as np
from stable_baselines3.common import noise

from yacht import Mode
from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import SampleAssetDataset
from yacht.environments.action_noises import build_action_noise


class ActionSchema(ABC):
    def __init__(
            self,
            num_assets: int,
            action_noise: Optional[noise.ActionNoise] = None
    ):
        self.num_assets = num_assets
        self.apply_noise = action_noise is not None
        if self.apply_noise:
            self.action_noise = noise.VectorizedActionNoise(base_noise=action_noise, n_envs=num_assets)
        else:
            self.action_noise = None

    @abstractmethod
    def get_action_space(self) -> Space:
        pass

    @abstractmethod
    def get_value(self, action: np.ndarray) -> np.ndarray:
        pass

    def reset(self):
        if self.apply_noise:
            self.action_noise.reset()


class DiscreteActionScheme(ActionSchema):
    def __init__(
            self,
            num_assets: int,
            possibilities: List[float],
            action_noise: Optional[noise.ActionNoise] = None
    ):
        super().__init__(num_assets=num_assets, action_noise=action_noise)

        self.possibilities = np.array(possibilities)

    def get_action_space(self) -> Space:
        return spaces.MultiDiscrete(
            nvec=[len(self.possibilities) for _ in range(self.num_assets)]

        )

    def get_value(self, action: np.ndarray) -> np.ndarray:
        action = self.possibilities[action]
        if self.apply_noise:
            action += self.action_noise()
            action = np.clip(action, a_min=np.min(self.possibilities), a_max=np.max(self.possibilities))

        return action


class ContinuousFloatActionSchema(ActionSchema):
    def __init__(
            self,
            num_assets: int,
            action_scaling_factor: float,
            action_noise: Optional[noise.ActionNoise] = None
    ):
        super().__init__(num_assets=num_assets, action_noise=action_noise)

        self.action_scaling_factor = action_scaling_factor

    def get_action_space(self) -> Space:
        return spaces.Box(low=-1, high=1, shape=(self.num_assets,))

    def get_value(self, action: np.ndarray) -> np.ndarray:
        action = action * self.action_scaling_factor
        if self.apply_noise:
            action += self.action_noise()
            action = np.clip(action, a_min=-1, a_max=1)

        return action


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


def build_action_schema(config: Config, dataset: SampleAssetDataset, mode: Mode):
    env_config: EnvironmentConfig = config.environment
    action_schema_class = action_schema_registry[env_config.action_schema]

    apply_noise = mode.is_trainable() and config.environment.use_action_noise
    if apply_noise:
        action_noise = build_action_noise(config)
    else:
        action_noise = None
    if action_schema_class in (ContinuousIntegerActionSchema, ContinuousFloatActionSchema):
        assert env_config.action_scaling_factor > 0

        return action_schema_class(
            num_assets=dataset.num_assets,
            action_scaling_factor=env_config.action_scaling_factor,
            action_noise=action_noise
        )
    else:
        assert len(env_config.possibilities) > 0

        return action_schema_class(
            num_assets=dataset.num_assets,
            possibilities=list(env_config.possibilities),
            action_noise=action_noise
        )
