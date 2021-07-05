from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from scipy.stats import norm

from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import TradingDataset


class RewardSchema(ABC):
    @abstractmethod
    def calculate_reward(self, action: Union[int, float], dataset: TradingDataset, current_index: int) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass


class RewardSchemaAggregator(RewardSchema):
    def __init__(self, reward_schemas: List[RewardSchema]):
        self.reward_schemas = reward_schemas

    def calculate_reward(self, action: Union[int, float], dataset: TradingDataset, current_index: int) -> float:
        rewards = [
            reward_schema.calculate_reward(action, dataset, current_index)
            for reward_schema in self.reward_schemas
        ]

        return sum(rewards)

    def reset(self):
        for reward_schema in self.reward_schemas:
            reward_schema.reset()


class ScaledRewardSchema(RewardSchema, ABC):
    def __init__(self, reward_scaling: float):
        assert 0 < reward_scaling <= 1, '"reward_scaling" should be within (0, 1].'

        self.reward_scaling = reward_scaling


class PriceChangeRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: Union[int, float], dataset: TradingDataset, current_index: int) -> float:
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
        reward *= self.reward_scaling

        return reward

    def reset(self):
        pass


class LeaderBoardRewardSchema(RewardSchema):
    def __init__(self, max_score: int):
        self.max_score = max_score
        self.total_score = 0

    def reset(self):
        self.total_score = 0

    def calculate_reward(self, action: Union[int, float], dataset: TradingDataset, current_index: int) -> float:
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

        score = 1 if action_side == price_side else -1
        score *= action_magnitude
        self.total_score += score

        return self.score_to_reward(self.total_score)

    def score_to_reward(self, score: float):
        # In a normal distribution values can go 2-3 standard distribution away from the mean value
        sigma = self.max_score / 2.5
        # See at what percentile in the distribution is the given score situated.
        percentile = norm.cdf(score, loc=0, scale=sigma)
        reward = self.percentile_to_reward(percentile)

        return reward

    @classmethod
    def percentile_to_reward(cls, p: float) -> int:
        assert 0 <= p <= 1

        if p < 0.5:
            return 0
        elif p < 0.6:
            return 1
        elif p < 0.7:
            return 2
        elif p < 0.8:
            return 3
        elif p < 0.9:
            return 5
        elif p < 0.92:
            return 8
        elif p < 0.94:
            return 13
        elif p < 0.95:
            return 21
        elif p < 0.96:
            return 34
        elif p < 0.97:
            return 55
        elif p < 98.:
            return 89
        elif p < 99:
            return 143
        else:
            return 232


#######################################################################################################################


reward_schema_registry = {
    'PriceChangeRewardSchema': PriceChangeRewardSchema,
    'LeaderBoardRewardSchema': LeaderBoardRewardSchema
}


def build_reward_schema(config: Config, max_score: int):
    env_config: EnvironmentConfig = config.environment
    reward_schemas = []
    for reward_schema_name in env_config.reward_schemas:
        reward_schema_class = reward_schema_registry[reward_schema_name]
        if reward_schema_class == PriceChangeRewardSchema:
            reward_schemas.append(
                PriceChangeRewardSchema(reward_scaling=env_config.reward_scaling)
            )
        elif reward_schema_class == LeaderBoardRewardSchema:
            reward_schemas.append(
                LeaderBoardRewardSchema(max_score=max_score)
            )

    return RewardSchemaAggregator(
        reward_schemas=reward_schemas
    )
