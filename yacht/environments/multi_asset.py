from typing import Dict, Optional

import numpy as np
import pandas as pd
from gym import spaces

from yacht.data.datasets import ChooseAssetDataset
from yacht.environments import BaseAssetEnvironment, RewardSchema, ActionSchema


class MultiAssetEnvironment(BaseAssetEnvironment):
    def __init__(
            self,
            name: str,
            dataset: ChooseAssetDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            seed: int = 0,
            render_on_done: bool = False,
            **kwargs
    ):
        super().__init__(name, dataset, reward_schema, action_schema, seed, render_on_done)

        self.buy_commission = kwargs.get('buy_commission', 0)
        self.sell_commission = kwargs.get('sell_commission', 0)

        # Internal state.
        self._initial_cash_position = kwargs['initial_cash_position']
        self._total_cash = self._initial_cash_position
        self._total_units = self._initialize_total_units(self.dataset)
        self._total_loss_commissions = 0

    def _reset(self):
        # Reset internal state.
        self._total_cash = self._initial_cash_position
        self._total_units = self._initialize_total_units(self.dataset)
        self._total_loss_commissions = 0

    @classmethod
    def _initialize_total_units(cls, dataset: ChooseAssetDataset) -> pd.Series:
        total_units = pd.Series(
            data=[0] * dataset.num_assets,
            index=dataset.asset_tickers,
            name='total_units',
            dtype=np.float32
        )

        return total_units

    def _get_observation_space(
            self,
            observation_space: Dict[str, Optional[spaces.Space]]
    ) -> Dict[str, Optional[spaces.Space]]:
        observation_space['env_features'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 1 + self.dataset.num_assets)
        )

        return observation_space

    def _get_next_observation(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        assert self.is_history_initialized

        # The slicing method does not work for window_size == 1.
        if self.window_size > 1:
            past_total_cash = self.history['total_cash'][-(self.window_size - 1):]
            past_total_units = self.history['total_units'][-(self.window_size - 1):]
        else:
            past_total_cash = []
            past_total_units = []

        total_cash_positions = past_total_cash + [self._total_cash]
        total_units = past_total_units + [self._total_units]
        total_cash_positions = np.array(total_cash_positions, dtype=np.float32).reshape(-1, 1)
        total_units = np.array(total_units, dtype=np.float32).reshape(-1, self.dataset.num_assets)

        observation['env_features'] = np.concatenate([total_cash_positions, total_units], axis=-1)

        return observation

    def scale_env_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return observation

    def _inverse_scaling(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        return observation

    def _initialize_history(self, history: dict) -> dict:
        history['total_units'] = (self.window_size - 1) * [self._total_units.values]
        history['total_assets'] = (self.window_size - 1) * [self._initial_cash_position]

        return history

    def _create_info(self, info: dict) -> dict:
        asset_prices = self.dataset.get_decision_prices(t_tick=self.t_tick)
        total_units_price = self._total_units.combine(
            other=asset_prices,
            func=lambda num_units, unit_price: num_units * unit_price
        )
        total_units_price = total_units_price.sum()

        info['total_loss_commissions'] = self._total_loss_commissions
        info['total_units'] = np.copy(self._total_units.values)
        info['total_assets'] = total_units_price + self._total_cash

        return info

    def update_internal_state(self, action: np.ndarray) -> dict:
        action = pd.Series(data=action, index=self.dataset.asset_tickers, name='Actions')
        action = action.sort_values()

        buy_asset_actions = action[action > 0]
        # Buy from the most conviction items to the least.
        buy_asset_actions = buy_asset_actions.iloc[::-1]
        for ticker, ticker_action in buy_asset_actions.iteritems():
            self._buy_asset(
                ticker,
                ticker_action
            )

        # Sell from the most conviction items to the least.
        sell_asset_actions = action[action < 0]
        for ticker, ticker_action in sell_asset_actions.iteritems():
            self._sell_asset(
                ticker,
                ticker_action
            )

        return {
            'total_cash': self._total_cash,
            'total_units': np.copy(self._total_units.values)
        }

    def _sell_asset(self, ticker: str, num_units_to_sell: float):
        asset_price = self.dataset.get_decision_prices(self.t_tick, ticker).item()

        if asset_price > 0 and self._total_units[ticker] > 0:
            sell_num_shares = min(abs(num_units_to_sell), self._total_units[ticker])
            sell_amount = asset_price * sell_num_shares * (1 - self.sell_commission)

            # Update balance.
            self._total_cash += sell_amount
            self._total_units[ticker] -= sell_num_shares
            self._total_loss_commissions += asset_price * sell_num_shares * self.sell_commission

    def _buy_asset(self, ticker: str, num_units_to_buy: float):
        asset_price = self.dataset.get_decision_prices(self.t_tick, ticker).item()

        if asset_price > 0:
            available_amount = self._total_cash / asset_price
            buy_num_shares = min(available_amount, num_units_to_buy)
            buy_amount = asset_price * buy_num_shares * (1 + self.buy_commission)

            # Update balance.
            self._total_cash -= buy_amount
            self._total_units[ticker] += buy_num_shares
            self._total_loss_commissions += asset_price * buy_num_shares * self.buy_commission

    def _is_done(self) -> bool:
        # If the agent has no more assets finish the episode.
        return self._total_cash <= 0 or (self._total_units <= 0).all()

    def render(self, mode='human', name='trades.png'):
        pass
