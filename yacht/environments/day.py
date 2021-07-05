import logging
from typing import Dict

import numpy as np
from gym import spaces
from yacht.environments import TradingEnv, Position
from yacht.environments.reward_schemas import LeaderBoardRewardSchema

logger = logging.getLogger(__file__)


class DayForecastEnv(TradingEnv):
    def reset(self):
        observation = super().reset()
        self._total_profit = 0.

        return observation

    def update_total_value(self, action):
        # TODO: Find a better way to calculate total_value & not duplicate code.
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

    def update_history(self, info):
        super().update_history(info)

        self.history['max_value'] = self._current_tick * self.action_schema.max_units_per_asset

    def max_possible_profit(self, stateless=True):
        if stateless:
            return self.action_schema.max_units_per_asset * (self._end_tick - self._start_tick)
        else:
            current_tick = self._start_tick
            last_trade_tick = current_tick
            total_num_ticks = self._end_tick - current_tick
            profit = 0.
            self.history['position'] = []
            self.history['action'] = []

            logger.info(f'A total of {total_num_ticks} ticks.')
            while current_tick + 1 <= self._end_tick:
                if self.prices[current_tick] <= self.prices[current_tick + 1]:
                    self.history['position'].append(Position.Long)
                    self.history['action'].append(self.action_schema.max_units_per_asset)
                    current_tick += 1

                    while current_tick + 1 <= self._end_tick and \
                            self.prices[current_tick] <= self.prices[current_tick + 1]:
                        current_tick += 1
                        self.history['position'].append(None)
                        self.history['action'].append(self.action_schema.max_units_per_asset)
                else:
                    self.history['position'].append(Position.Short)
                    self.history['action'].append(-self.action_schema.max_units_per_asset)
                    current_tick += 1

                    while current_tick + 1 <= self._end_tick and \
                            self.prices[current_tick] > self.prices[current_tick + 1]:
                        current_tick += 1
                        self.history['position'].append(None)
                        self.history['action'].append(-self.action_schema.max_units_per_asset)

                num_ticks = current_tick - last_trade_tick
                profit += num_ticks

                last_trade_tick = current_tick

            profit = profit * self.action_schema.max_units_per_asset

            logger.info(f'{profit}/{total_num_ticks} accuracy.')
            self._total_value = profit

            return profit
