from typing import Dict, Optional

import numpy as np
from gym import spaces

from yacht.data.datasets import TradingDataset
from yacht.environments import TradingEnv, RewardSchema, ActionSchema


class SingleAssetTradingEnvironment(TradingEnv):
    def __init__(
            self,
            dataset: TradingDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            seed: int = 0,
            **kwargs
    ):
        super().__init__(dataset, reward_schema, action_schema, seed)

        self.buy_commission = kwargs.get('buy_commission', 0)
        self.sell_commission = kwargs.get('sell_commission', 0)

        # Internal state.
        self._initial_cash_position = kwargs['initial_cash_position']
        self._total_value = self._initial_cash_position
        self._total_units = 0
        self._total_loss_commissions = 0

    def _reset(self):
        # Reset global information.
        self._total_value = self._initial_cash_position
        self._total_units = 0
        self._total_loss_commissions = 0

    def _get_observation_space(
            self,
            observation_space: Dict[str, Optional[spaces.Space]]
    ) -> Dict[str, Optional[spaces.Space]]:
        observation_space['env_features'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, 2, ))

        return observation_space

    def _get_next_observation(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        # The slicing method does not work for window_size == 1, therefore the else branch will create and empty list.
        if self.is_history_initialized and self.window_size > 1:
            past_total_values = self.history['total_value'][-(self.window_size - 1):]
            past_total_units = self.history['total_units'][-(self.window_size - 1):]
        else:
            past_total_values = (self.window_size - 1) * [self._initial_cash_position]
            past_total_units = (self.window_size - 1) * [0]
        total_values = past_total_values + [self._total_value]
        total_units = past_total_units + [self._total_units]
        total_values = np.array(total_values, dtype=np.float32).reshape(-1, 1)
        total_units = np.array(total_units, dtype=np.float32).reshape(-1, 1)

        observation['env_features'] = np.concatenate([total_values, total_units], axis=-1)

        return observation

    def _initialize_history(self, history: dict) -> dict:
        history['total_units'] = (self.window_size - 1) * [self._total_units]
        history['total_assets'] = (self.window_size - 1) * [self._initial_cash_position]

        return history

    def _create_info(self, info: dict) -> dict:
        info['total_units'] = self._total_units
        info['total_assets'] = self._s_t['env_features'][-1][0] + \
            self._s_t['env_features'][-1][1] * self._s_t['1d'][-1, 0, 0]

        return info

    def update_internal_state(self):
        a_t = self._a_t.item()
        if a_t > 0:
            self._buy_asset(a_t)
        elif a_t < 0:
            self._sell_asset(a_t)

    def _sell_asset(self, action: float):
        # Take the close price from the '1d' frequency & the last window.
        stock_close_price = self._s_t['1d'][-1, 0, 0]

        # Sell only if the price is valid and current asset is > 0.
        if stock_close_price > 0 and self._total_units > 0:
            sell_num_shares = min(abs(action), self._total_units)
            sell_amount = stock_close_price * sell_num_shares * (1 - self.sell_commission)

            # Update balance.
            self._total_value += sell_amount
            self._total_units -= sell_num_shares
            self._total_loss_commissions += stock_close_price * sell_num_shares * self.sell_commission
        else:
            sell_num_shares = 0

        return sell_num_shares

    def _buy_asset(self, action: float):
        # Take the close price from the '1d' frequency & the last window.
        stock_close_price = self._s_t['1d'][-1, 0, 0]

        # Buy only if the price is > 0.
        if stock_close_price > 0:
            available_amount = self._total_value / stock_close_price
            buy_num_shares = min(available_amount, action)
            buy_amount = stock_close_price * buy_num_shares * (1 + self.buy_commission)

            # Update balance.
            self._total_value -= buy_amount
            self._total_units += buy_num_shares
            self._total_loss_commissions += stock_close_price * buy_num_shares * self.buy_commission
        else:
            buy_num_shares = 0

        return buy_num_shares

    def _on_done(self) -> bool:
        # If the agent has no more assets finish the episode.
        return self._total_value <= 0 and self._total_units <= 0

    def render(self, mode='human', name='trades.png'):
        pass

    def max_possible_profit(self, stateless: bool = True) -> float:
        pass
