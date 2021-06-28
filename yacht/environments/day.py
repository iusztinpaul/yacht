import logging

import numpy as np

from yacht.environments import TradingEnv, Position
from yacht.environments.reward_schemas import DayTotalValueRewardSchema

logger = logging.getLogger(__file__)


class DayForecastEnv(TradingEnv):
    def reset(self):
        observation = super().reset()
        self._total_profit = 0.

        return observation

    def update_total_value(self, action):
        if isinstance(self.reward_schema, DayTotalValueRewardSchema):
            self._total_value = self.reward_schema.total_value
        else:
            raise NotImplementedError()

    def get_next_observation(self) -> np.array:
        observation = super().get_next_observation()

        total_value = np.tile(self._total_value, (self.window_size, observation.shape[1], 1))
        observation = np.concatenate([observation, total_value], axis=-1)

        return observation

    def max_possible_profit(self):
        # FIXME: obsolete

        current_tick = self._start_tick
        last_trade_tick = current_tick
        total_num_ticks = self._end_tick - current_tick
        profit = 0.

        logger.info(f'A total of {total_num_ticks} ticks.')
        while current_tick + 1 <= self._end_tick:
            if self.prices[current_tick] <= self.prices[current_tick + 1]:
                self._position_history.append(Position.Long)
                current_tick += 1

                while current_tick + 1 <= self._end_tick and \
                        self.prices[current_tick] <= self.prices[current_tick + 1]:
                    current_tick += 1
                    self._position_history.append(None)
            else:
                self._position_history.append(Position.Short)
                current_tick += 1

                while current_tick + 1 <= self._end_tick and \
                        self.prices[current_tick] > self.prices[current_tick + 1]:
                    current_tick += 1
                    self._position_history.append(None)

            num_ticks = current_tick - last_trade_tick
            profit += num_ticks

            last_trade_tick = current_tick

        logger.info(f'{profit}/{total_num_ticks} accuracy.')
        self._total_reward = profit
        self._total_profit = profit

        return profit
