from abc import ABC, abstractmethod
from typing import List

from gym import Space, spaces
import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise

from yacht import Mode
from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import SampleAssetDataset


class ActionSchema(ABC):
    def __init__(
            self,
            num_assets: int,
            apply_noise: bool = False,
            mean: float = None,
            sigma: float = None
    ):
        self.num_assets = num_assets
        self.apply_noise = apply_noise
        if self.apply_noise:
            assert mean is not None
            assert sigma is not None

            self.noise = VectorizedActionNoise(
                base_noise=OrnsteinUhlenbeckActionNoise(
                    mean=np.array(mean, dtype=np.float32),
                    sigma=np.array(sigma, dtype=np.float32)
                ),
                n_envs=num_assets
            )
        else:
            self.noise = None

    @abstractmethod
    def get_action_space(self) -> Space:
        pass

    @abstractmethod
    def get_value(self, action: np.ndarray) -> np.ndarray:
        pass

    def reset(self):
        if self.apply_noise:
            self.noise.reset()


class DiscreteActionScheme(ActionSchema):
    def __init__(
            self,
            num_assets: int,
            possibilities: List[float],
            apply_noise: bool = False,
            mean: float = None,
            sigma: float = None
    ):
        super().__init__(num_assets=num_assets, apply_noise=apply_noise, mean=mean, sigma=sigma)

        self.possibilities = np.array(possibilities)

    def get_action_space(self) -> Space:
        return spaces.MultiDiscrete(
            nvec=[len(self.possibilities) for _ in range(self.num_assets)]

        )

    def get_value(self, action: np.ndarray) -> np.ndarray:
        action = self.possibilities[action]
        if self.apply_noise:
            action += self.noise()
            action = np.clip(action, a_min=np.min(self.possibilities), a_max=np.max(self.possibilities))

        return action


class ContinuousFloatActionSchema(ActionSchema):
    def __init__(
            self,
            num_assets: int,
            action_scaling_factor: float,
            apply_noise: bool = False,
            mean: float = None,
            sigma: float = None
    ):
        super().__init__(num_assets=num_assets, apply_noise=apply_noise, mean=mean, sigma=sigma)

        self.action_scaling_factor = action_scaling_factor

    def get_action_space(self) -> Space:
        return spaces.Box(low=-1, high=1, shape=(self.num_assets,))

    def get_value(self, action: np.ndarray) -> np.ndarray:
        action = action * self.action_scaling_factor
        if self.apply_noise:
            action += self.noise()
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
    if action_schema_class in (ContinuousIntegerActionSchema, ContinuousFloatActionSchema):
        assert env_config.action_scaling_factor > 0

        return action_schema_class(
            num_assets=dataset.num_assets,
            action_scaling_factor=env_config.action_scaling_factor,
            apply_noise=apply_noise,
            mean=config.environment.action_noise_mean,
            sigma=config.environment.action_noise_sigma,
        )
    else:
        assert len(env_config.possibilities) > 0

        return action_schema_class(
            num_assets=dataset.num_assets,
            possibilities=list(env_config.possibilities),
            apply_noise=apply_noise,
            mean=config.environment.action_noise_mean,
            sigma=config.environment.action_noise_sigma,
        )
