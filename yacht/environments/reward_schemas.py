import warnings
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from scipy.stats import norm

from yacht import utils
from yacht.config import Config
from yacht.config.proto.environment_pb2 import EnvironmentConfig
from yacht.data.datasets import AssetDataset

#########################################################
#################### INTERFACES #########################
#########################################################
from yacht.utils import build_from_protobuf


class RewardSchema(ABC):
    @abstractmethod
    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        pass


class ScaledRewardSchema(RewardSchema, ABC):
    def __init__(self, reward_scaling: float):
        assert 0 < reward_scaling, '"reward_scaling" should be positive.'

        self.reward_scaling = reward_scaling


class RewardSchemaAggregator(ScaledRewardSchema):
    def __init__(self, reward_schemas: List[RewardSchema], reward_scaling: float):
        super().__init__(reward_scaling=reward_scaling)

        self.reward_schemas = reward_schemas

    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        rewards = [
            reward_schema.calculate_reward(action, *args, **kwargs)
            for reward_schema in self.reward_schemas
        ]
        rewards = sum(rewards)
        rewards *= self.reward_scaling

        return rewards


######################################################
#################### TRADING #########################
######################################################


class AssetsPriceChangeRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        current_state = kwargs['current_state']
        next_state = kwargs['next_state']

        begin_total_assets = current_state['env_features'][-1][0] + \
                             (current_state['env_features'][-1][1:] * current_state['1d'][-1, 0, :, 1]).sum()
        end_total_assets = next_state['env_features'][-1][0] + \
                           (next_state['env_features'][-1][1:] * next_state['1d'][-1, 0, :, 1]).sum()

        reward = end_total_assets - begin_total_assets
        reward = reward * self.reward_scaling

        return reward


##############################################################
#################### ORDER EXECUTION #########################
##############################################################


class PriceAdvantageRelativeToCashPositionRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        market_mean_price = kwargs['market_mean_price']
        next_price = kwargs['next_price']
        initial_cash_position = kwargs['initial_cash_position']
        remained_cash = kwargs['remained_cash']

        price_advantage = (1 - next_price / market_mean_price)
        remained_cash_ratio = remained_cash / initial_cash_position
        # Map ratio to [-1; 1]
        remained_cash_ratio *= 2
        remained_cash_ratio -= 1

        reward = price_advantage * remained_cash_ratio
        reward *= self.reward_scaling

        return reward.item()


class DecisionMakingRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: np.ndarray, *args, **kwargs):
        # TODO: Adapt for sell execution
        market_mean_price = kwargs['market_mean_price']
        next_price = kwargs['next_price']

        price_advantage = (1 - next_price / market_mean_price)
        reward = self.reward_scaling * action * price_advantage

        return reward.item()


class ActionMagnitudeRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        reward = action ** 2
        reward = -reward * self.reward_scaling

        return reward.item()


class ActionDistanceRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        actions = np.array(kwargs['actions'], dtype=np.float32)
        max_distance = kwargs['max_distance']

        action_indices = np.where(actions != 0)[0]
        num_actions = action_indices.shape[0]
        if num_actions <= 1:
            return 0.

        # TODO: Adapt for multi-assets.
        # Compute the absolute mean difference between an action and every action before it.
        action_indices = np.tile(action_indices, reps=(num_actions, 1))
        diag_action_indices = np.expand_dims(np.diag(action_indices), axis=1)
        difference = action_indices - diag_action_indices
        difference = np.tril(difference)
        difference = difference[difference != 0]
        difference += 1  # Don't reward adjacent actions.
        difference = np.abs(difference)
        difference = np.mean(difference)

        reward = difference / (max_distance + 1)
        reward *= action
        reward *= self.reward_scaling

        return reward.item()


class CashOnLastTickRewardSchema(ScaledRewardSchema):
    def calculate_reward(self, action: np.ndarray, *args, **kwargs) -> float:
        actions = kwargs['actions']
        max_distance = kwargs['max_distance']
        # This reward should be given only in the end.
        if len(actions) < max_distance:
            return 0.

        cash_used = kwargs['cash_used_on_last_tick']
        initial_cash_position = kwargs['initial_cash_position']

        reward = cash_used / initial_cash_position
        reward = -reward * self.reward_scaling

        return reward


###########################################################
#################### SCORE BASED  #########################
############################################################


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
    'PriceAdvantageRelativeToCashPositionRewardSchema': PriceAdvantageRelativeToCashPositionRewardSchema,
    'AssetsPriceChangeRewardSchema': AssetsPriceChangeRewardSchema,
    'DecisionMakingRewardSchema': DecisionMakingRewardSchema,
    'ActionMagnitudeRewardSchema': ActionMagnitudeRewardSchema,
    'ActionDistanceRewardSchema': ActionDistanceRewardSchema,
    'CashOnLastTickRewardSchema': CashOnLastTickRewardSchema,
    'LeaderBoardRewardSchema': LeaderBoardRewardSchema
}


def build_reward_schema(config: Config):
    env_config: EnvironmentConfig = config.environment
    reward_schemas: List[RewardSchema] = []
    for reward_schema_config in env_config.reward_schemas:
        reward_schema_class = reward_schema_registry[reward_schema_config.name]
        reward_schema: RewardSchema = build_from_protobuf(reward_schema_class, reward_schema_config)

        reward_schemas.append(reward_schema)

    if env_config.global_reward_scaling == 0:
        warnings.warn(
            '"config.env.global_reward_scaling=0" -> it will set all rewards to 0. '
            'Because of this we will force it to be equal to "=1".'
        )
        global_reward_scaling = 1
    else:
        global_reward_scaling = env_config.global_reward_scaling

    return RewardSchemaAggregator(
        reward_schemas=reward_schemas,
        reward_scaling=global_reward_scaling
    )
