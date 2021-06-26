from abc import ABC
from typing import Union

import numpy as np
import pandas as pd

from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import TradingDataset


class RewardSchema(ABC):
    def calculate_reward(self, action: Union[int, float], dataset: TradingDataset, current_index: int):
        raise NotImplementedError()


class DayTotalValueRewardSchema(RewardSchema):
    def __init__(self, reward_scaling: float):
        assert 0 < reward_scaling <= 1, '"reward_scaling" should be within (0, 1].'

        self.total_value = 0
        self.reward_scaling = reward_scaling

    def calculate_reward(self, action: np.array, dataset: TradingDataset, current_index: int):
        # TODO: Make this function to support multiple asset actions
        assert action.shape[0] == 1 and len(action.shape) == 1
        action = action.item()

        # TODO: Find a better way to get the future close price.
        #  For now only accessing 'data' we have data removed at k folding, but it is now clean at all.
        if hasattr(dataset, 'getitem_index_mappings'):
            current_index = dataset.getitem_index_mappings[current_index]
        current_close_price = dataset.data['1d'].iloc[current_index]['Close']
        future_close_price = dataset.data['1d'].iloc[current_index + 1]['Close']

        action_side = np.sign(action)
        action_magnitude = np.abs(action)
        if action_side > 0:
            if future_close_price > current_close_price:
                next_value = self.total_value + action_magnitude
            else:
                next_value = self.total_value - action_magnitude
        elif action_side < 0:
            if future_close_price < current_close_price:
                next_value = self.total_value + action_magnitude
            else:
                next_value = self.total_value - action_magnitude
        else:
            next_value = self.total_value

        reward = (next_value - self.total_value) * self.reward_scaling
        self.total_value = next_value

        return reward


#######################################################################################################################


reward_schema_registry = {
    'DayTotalValueRewardSchema': DayTotalValueRewardSchema
}


def build_reward_schema(env_config: EnvironmentConfig):
    reward_schema_class = reward_schema_registry[env_config.reward_schema]

    return reward_schema_class(
        reward_scaling=env_config.reward_scaling
    )
