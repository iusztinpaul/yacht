import logging
from typing import Dict

import numpy as np
import pandas as pd

from gym import spaces
from yacht.environments import BaseAssetEnvironment, Position
from yacht.environments.reward_schemas import LeaderBoardRewardSchema

logger = logging.getLogger(__file__)


class DayForecastEnvironment(BaseAssetEnvironment):
    def update_internal_state(self):
        # TODO: Find a better way to calculate total_cash & not duplicate code.
        leader_board_reward_schema = [
            reward_schema for reward_schema in self.reward_schema.reward_schemas
            if isinstance(reward_schema, LeaderBoardRewardSchema)
        ][0]

        self._total_value = leader_board_reward_schema.total_score

    def get_observation_space(self) -> Dict[str, spaces.Space]:
        observation_space = super().get_observation_space()
        observation_space['env_features'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

        return observation_space

    def get_next_observation(self) -> Dict[str, np.array]:
        observation = super().get_next_observation()
        observation['env_features'] = self._total_value

        return observation

    def create_info(self) -> dict:
        info = super().create_info()

        info['max_possible_value'] = (self.t_tick - self.start_tick) * self.action_schema.action_scaling_factor
        info['total_assets'] = round(self._total_value / self.max_possible_profit(stateless=True), 2)

        return info

    def create_baseline_report(self) -> pd.DataFrame:
        # Clean env.
        self.reset()
        # Populate the history attribute.
        self.max_possible_profit(stateless=False)

        # Create report based on the previous populated history attribute.
        return self.create_report()

    def max_possible_profit(self, stateless=True):
        if stateless:
            return self.action_schema.action_scaling_factor * (self.end_tick - self.start_tick)
        else:
            current_tick = self.start_tick
            total_num_ticks = self.end_tick - current_tick
            total_value = 0.

            prices = self.prices['Close'].values

            self.history['position'] = []
            self.history['action'] = []
            self.history['total_value'] = []
            self.history['date'] = []

            logger.info(f'A total of {total_num_ticks} ticks.')
            while current_tick + 1 <= self.end_tick:
                if prices[current_tick] <= prices[current_tick + 1]:
                    total_value += self.action_schema.action_scaling_factor

                    self.history['position'].append(Position.Long)
                    self.history['action'].append(self.action_schema.action_scaling_factor)
                    self.history['total_value'].append(total_value)
                    self.history['date'].append(self.dataset.index_to_datetime(current_tick))

                    current_tick += 1

                    while current_tick + 1 <= self.end_tick and \
                            prices[current_tick] <= prices[current_tick + 1]:
                        total_value += self.action_schema.action_scaling_factor

                        self.history['position'].append(None)
                        self.history['action'].append(self.action_schema.action_scaling_factor)
                        self.history['total_value'].append(total_value)
                        self.history['date'].append(self.dataset.index_to_datetime(current_tick))

                        current_tick += 1

                else:
                    total_value += self.action_schema.action_scaling_factor

                    self.history['position'].append(Position.Short)
                    self.history['action'].append(-self.action_schema.action_scaling_factor)
                    self.history['total_value'].append(total_value)
                    self.history['date'].append(self.dataset.index_to_datetime(current_tick))

                    current_tick += 1

                    while current_tick + 1 <= self.end_tick and \
                            prices[current_tick] > prices[current_tick + 1]:
                        total_value += self.action_schema.action_scaling_factor

                        self.history['position'].append(None)
                        self.history['action'].append(-self.action_schema.action_scaling_factor)
                        self.history['total_value'].append(total_value)
                        self.history['date'].append(self.dataset.index_to_datetime(current_tick))

                        current_tick += 1

            logger.info(f'{total_value}/{total_num_ticks} accuracy.')

            return total_value
