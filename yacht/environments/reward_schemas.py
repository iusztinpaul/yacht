import inspect
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from scipy.stats import norm

from yacht import utils
from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import AssetDataset


class RewardSchema(ABC):
    @abstractmethod
    def calculate_reward(self, action: np.ndarray, current_state: dict, next_state: dict) -> float:
        pass


class RewardSchemaAggregator(RewardSchema):
    def __init__(self, reward_schemas: List[RewardSchema]):
        self.reward_schemas = reward_schemas

    def calculate_reward(self, action: np.ndarray, current_state: dict, next_state: dict) -> float:
        rewards = [
            reward_schema.calculate_reward(action, current_state, next_state)
            for reward_schema in self.reward_schemas
        ]

        return sum(rewards)


class ScaledRewardSchema(RewardSchema, ABC):
    def __init__(self, reward_scaling: float):
        assert 0 < reward_scaling <= 1, '"reward_scaling" should be within (0, 1].'

        self.reward_scaling = reward_scaling


class AssetsPriceChangeRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: Union[int, float], current_state: dict, next_state: dict) -> float:
        begin_total_assets = current_state['env_features'][-1][0] + \
            (current_state['env_features'][-1][1:] * current_state['1d'][-1, 0, :, 1]).sum()
        end_total_assets = next_state['env_features'][-1][0] + \
            (next_state['env_features'][-1][1:] * next_state['1d'][-1, 0, :, 1]).sum()

        reward = end_total_assets - begin_total_assets
        reward = reward * self.reward_scaling

        return reward


class ActionMagnitudeRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: Union[int, float], current_state: dict, next_state: dict) -> float:
        reward = action ** 2
        reward = reward.sum()
        reward = -reward * self.reward_scaling

        return reward


class ScoreBasedRewardSchema(RewardSchema, ABC):
    def calculate_reward(self, action: Union[int, float], dataset: AssetDataset, current_index: int) -> float:
        # TODO: Make this function to support multiple asset actions
        assert action.shape[0] == 1 and len(action.shape) == 1
        action = action.item()

        # TODO: Find a better way to get the future close price.
        #  For now only accessing 'data' we have data removed at k folding, but it is not clean at all.
        if hasattr(dataset, 'getitem_index_mappings'):
            current_index = dataset.getitem_index_mappings[current_index]
        current_close_price = dataset.data['1d'].iloc[current_index]['Close']
        future_close_price = dataset.data['1d'].iloc[current_index + 1]['Close']

        action_magnitude = np.abs(action)
        action_side = np.sign(action)
        price_side = np.sign(future_close_price - current_close_price)

        reward = 1 if action_side == price_side else -1
        reward *= action_magnitude

        return reward


class PriceChangeRewardSchema(ScoreBasedRewardSchema):
    def __init__(self, reward_scaling: float):
        assert 0 < reward_scaling <= 1, '"reward_scaling" should be within (0, 1].'

        self.reward_scaling = reward_scaling

    def calculate_reward(self, action: Union[int, float], dataset: AssetDataset, current_index: int) -> float:
        reward = super().calculate_reward(action, dataset, current_index)
        reward *= self.reward_scaling

        return reward


class LeaderBoardRewardSchema(ScoreBasedRewardSchema):
    def __init__(self, max_score: int, density_thresholds: List[float]):
        self.max_score = max_score
        # TODO: Get the total_score state from the env.
        self.total_score = 0

        self.density_thresholds = density_thresholds
        assert len(self.density_thresholds) % 2 == 0

        # Rewards are symmetric relative to 0.5
        self.thresholds_cutting_point = len(self.density_thresholds) // 2
        self.rewards = utils.fib_sequence(
            n=self.thresholds_cutting_point + 1
        )

        assert self.density_thresholds[self.thresholds_cutting_point] == 0.5
        assert self.thresholds_cutting_point + 1 == len(self.rewards)

    def reset(self):
        self.total_score = 0

    def calculate_reward(self, action: Union[int, float], dataset: AssetDataset, current_index: int) -> float:
        score = super().calculate_reward(action, dataset, current_index)
        self.total_score += score

        return self.score_to_reward(self.total_score)

    def score_to_reward(self, score: float):
        # In a normal distribution values can go 2-3 standard distribution away from the mean value.
        # We make sigma only 2.5 to increase the chance of the agent to reach one of the extremes.
        sigma = self.max_score / 2.5
        # See what percentile of the density distribution is behind 'score' value.
        percentile = norm.cdf(score, loc=0, scale=sigma)
        reward = self.cdf_density_percentile_to_reward(percentile)

        return reward

    def cdf_density_percentile_to_reward(self, p: float) -> int:
        assert 0 <= p <= 1

        if p == 0.5:
            return 0
        elif p < 0.5:
            return -self._associate_percentile_with_reward(
                thresholds=self.density_thresholds[:self.thresholds_cutting_point][::-1],
                rewards=self.rewards,
                p=p
            )
        else:
            return self._associate_percentile_with_reward(
                thresholds=self.density_thresholds[self.thresholds_cutting_point:],
                rewards=self.rewards,
                p=p
            )

    @classmethod
    def _associate_percentile_with_reward(cls, thresholds: List[float], rewards: List[int], p: float) -> int:
        for i in range(len(rewards)):
            if p <= thresholds[i]:
                return rewards[i]
        else:
            return rewards[i + 1]


#######################################################################################################################


reward_schema_registry = {
    'AssetsPriceChangeRewardSchema': AssetsPriceChangeRewardSchema,
    'ActionMagnitudeRewardSchema': ActionMagnitudeRewardSchema,
    'PriceChangeRewardSchema': PriceChangeRewardSchema,
    'LeaderBoardRewardSchema': LeaderBoardRewardSchema
}


def build_reward_schema(config: Config):
    env_config: EnvironmentConfig = config.environment
    reward_schemas = []
    for reward_schema_config in env_config.reward_schemas:
        reward_schema_class = reward_schema_registry[reward_schema_config.name]

        # Create kwargs for specific class.
        possible_class_kwargs = {
            'reward_scaling': reward_schema_config.reward_scaling,
            'density_thresholds': list(reward_schema_config.density_thresholds)
        }
        class_signature = inspect.signature(reward_schema_class.__init__)
        class_constructor_parameters = class_signature.parameters.keys()
        class_kwargs = {}
        for k, v in possible_class_kwargs.items():
            if k in class_constructor_parameters:
                class_kwargs[k] = v

        reward_schemas.append(
            reward_schema_class(**class_kwargs)
        )

    return RewardSchemaAggregator(
        reward_schemas=reward_schemas
    )
