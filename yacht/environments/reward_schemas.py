from abc import ABC
from typing import Union

import numpy as np
import pandas as pd

from yacht.config.proto.environment_pb2 import EnvironmentConfig


class RewardSchema(ABC):
    def calculate_reward(self, action: Union[int, float], prices: pd.DataFrame, action_tick: int):
        raise NotImplementedError()


class DayCurrentValueRewardSchema(RewardSchema):
    def __init__(self, reward_scaling: float):
        assert 0 < reward_scaling <= 1, '"reward_scaling" should be within (0, 1].'

        self.current_value = 0
        self.reward_scaling = reward_scaling

    def calculate_reward(self, action: np.array, prices: pd.DataFrame, action_tick: int):
        # TODO: Make this function to support multiple asset actions
        assert action.shape[0] == 1 and len(action.shape) == 1
        action = action.item()

        current_price = prices[action_tick]
        future_price = prices[action_tick + 1]

        prediction_sign = np.sign(action)
        if prediction_sign > 0:
            if future_price > current_price:
                next_value = self.current_value + np.abs(action)
            else:
                next_value = self.current_value + np.abs(action)
        elif prediction_sign < 0:
            if future_price < current_price:
                next_value = self.current_value + np.abs(action)
            else:
                next_value = self.current_value + np.abs(action)
        else:
            next_value = self.current_value

        reward = (next_value - self.current_value) * self.reward_scaling
        self.current_value = next_value

        return reward


#######################################################################################################################


reward_schema_registry = {
    'DayCurrentValueRewardSchema': DayCurrentValueRewardSchema
}


def build_reward_schema(env_config: EnvironmentConfig):
    reward_schema_class = reward_schema_registry[env_config.reward_schema]

    return reward_schema_class(
        reward_scaling=env_config.reward_scaling
    )
