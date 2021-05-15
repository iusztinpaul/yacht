import numpy as np

from yacht.environments import TradingEnv, Positions


class DayForecastEnv(TradingEnv):
    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price

        if self._position == Positions.Short:
            sign = 1 if price_diff < 0 else -1
        elif self._position == Positions.Long:
            sign = 1 if price_diff > 0 else -1
        else:
            raise RuntimeError(f'Wrong Position: {self._position}')

        step_reward = sign * np.log10(np.abs(price_diff)) if price_diff != 0 else 0

        return step_reward

    def _update_profit(self, action):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._position == Positions.Short:
            if last_trade_price > current_price:
                self._total_profit += 1
            else:
                self._total_profit -= 1

        if self._position == Positions.Long:
            if current_price > last_trade_price:
                self._total_profit += 1
            else:
                self._total_profit -= 1
