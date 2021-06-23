import logging

from yacht.environments import TradingEnv, Position
from yacht.environments.reward_schemas import DayCurrentValueRewardSchema

logger = logging.getLogger(__file__)


class DayForecastEnv(TradingEnv):
    def reset(self):
        observation = super().reset()
        self._total_profit = 0.

        return observation

    def calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price

        if self._position == Position.Short:
            sign = 1 if price_diff < 0 else -1
        elif self._position == Position.Long:
            sign = 1 if price_diff > 0 else -1
        else:
            raise RuntimeError(f'Wrong Position: {self._position}')

        # step_reward = sign * np.log10(np.abs(price_diff)) if price_diff != 0 else 0
        step_reward = sign if price_diff != 0 else 0

        return step_reward

    def update_profit(self, action):
        if isinstance(self.reward_schema, DayCurrentValueRewardSchema):
            self._total_profit = self.reward_schema.current_value
        else:
            raise NotImplementedError()

    def max_possible_profit(self):
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
