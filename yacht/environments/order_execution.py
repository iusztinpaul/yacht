import os
from copy import copy
from typing import Optional, Dict

import numpy as np
import pandas as pd
from gym import spaces
from tqdm import tqdm

from yacht.data.datasets import SampleAssetDataset
from yacht.environments import RewardSchema, ActionSchema
from yacht.environments.multi_asset import MultiAssetEnvironment


class OrderExecutionEnvironment(MultiAssetEnvironment):
    def __init__(
            self,
            name: str,
            dataset: SampleAssetDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            compute_metrics: bool = False,
            add_action_features: bool = False,
            **kwargs
    ):
        self.add_action_features = add_action_features

        super().__init__(name, dataset, reward_schema, action_schema, compute_metrics, **kwargs)

        self.unadjusted_period_mean_price = None
        self.cash_used_on_last_tick = 0

    def _reset(self):
        super()._reset()

        # We care about the mean price only in the unadjusted range.
        self.unadjusted_period_mean_price = self.dataset.compute_mean_price(
            start=self.dataset.sampled_dataset.unadjusted_start,
            end=self.dataset.sampled_dataset.end
        )
        self.cash_used_on_last_tick = 0

    def _get_observation_space(
            self,
            observation_space: Dict[str, Optional[spaces.Space]]
    ) -> Dict[str, Optional[spaces.Space]]:
        observation_space['env_features'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 3 if self.add_action_features else 2)
        )

        return observation_space

    def _get_env_observation(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        assert self.is_history_initialized

        remained_cash_history = self.history['remained_cash'][-self.window_size:]
        remained_cash_history = np.array(remained_cash_history, dtype=np.float32).reshape(-1, 1)

        used_time_history = self.history['used_time'][-self.window_size:]
        used_time_history = np.array(used_time_history, dtype=np.float32).reshape(-1, 1)

        if self.add_action_features:
            actions = self.history['action'][-self.window_size:]
            actions = np.array(actions, dtype=np.float32).reshape(-1, 1)

            observation['env_features'] = np.concatenate([remained_cash_history, used_time_history, actions], axis=-1)
        else:
            observation['env_features'] = np.concatenate([remained_cash_history, used_time_history], axis=-1)

        # In the first iterations, in some setups, we don't have enough history.
        if observation['env_features'].shape[0] < self.window_size:
            if self.window_size == 1:
                assert self.add_action_features is False, 'window_size = 1 does not support add_action_features = True.'
                observation['env_features'] = np.array([[1., 0.]], dtype=np.float32)
            else:
                padding_value = self.window_size - observation['env_features'].shape[0]
                observation['env_features'] = np.pad(
                    observation['env_features'],
                    ((padding_value, 0), (0, 0)),
                    mode='edge'
                )

        return observation

    def _initialize_history(self, history: dict) -> dict:
        history = super()._initialize_history(history)

        history['remained_cash'] = self.period_adjustment_size * [1.]
        history['used_time'] = self.period_adjustment_size * [0.]

        return history

    def update_internal_state(self, action: np.ndarray) -> dict:
        changes = super().update_internal_state(action)
        changes['remained_cash'] = self._compute_total_cash_ratio()
        changes['used_time'] = self._compute_used_time_ratio()

        return changes

    def _buy_asset(self, ticker: str, cash_ratio_to_use: float):
        if self._total_cash > 0:
            buy_amount = cash_ratio_to_use * self._initial_cash_position
            if buy_amount > self._total_cash:
                buy_amount = self._total_cash
                action_index = np.where(self._total_units.index == ticker)[0]
                # Change the action to log the real value taken because of the lack of cash.
                # We change it here, because in _filter_actions it would be unnecessary computations to do this for
                # all the tickers.
                self._a_t[action_index] = self._total_cash / self._initial_cash_position

            decision_tick = self.get_decision_tick()
            asset_last_price = self.dataset.get_decision_prices(decision_tick, ticker)
            assert asset_last_price.notna().all(), 'Cannot buy assets with price = nan.'
            asset_last_price = asset_last_price.item()

            commission_amount = buy_amount * self.buy_commission
            asset_buy_amount = buy_amount - commission_amount
            buy_num_shares = asset_buy_amount / asset_last_price

            # Update balance.
            self._total_cash -= buy_amount
            self._total_units[ticker] += buy_num_shares
            self._total_loss_commissions += commission_amount

    def _adjust_actions(self, actions: np.ndarray) -> np.ndarray:
        if not self.has_cash():
            return np.zeros_like(actions)

        if self.is_done():
            # Make a copy to keep it for metrics.
            self.cash_used_on_last_tick = copy(self._total_cash)
            # Remove the cash that the agent actually tried to use. That is a valid move.
            self.cash_used_on_last_tick -= (actions * self._initial_cash_position).sum()
            # The buy & sell actions will handle the case when the agent tries to buy more than it has.
            # But in that case, put 'cash_used_on_last_tick' on 0.
            self.cash_used_on_last_tick = max(self.cash_used_on_last_tick, 0)
            self.cash_used_on_last_tick /= self._initial_cash_position

            # Split money equally between the assets.
            action_remaining_month_cash = np.tile(
                self.cash_used_on_last_tick // self.dataset.num_assets,
                self.dataset.num_assets
            )
            actions += action_remaining_month_cash

        return actions

    def _get_reward_schema_kwargs(self, next_state: Dict[str, np.ndarray]) -> dict:
        decision_tick = self.get_decision_tick()
        last_price = self.dataset.get_decision_prices(decision_tick).values

        return {
            'market_mean_price': self.unadjusted_period_mean_price.values,
            'last_price': last_price,
            'actions': self.history['action'][self.window_size:],
            'max_distance': self.dataset.num_days,
            'cash_used_on_last_tick': self.cash_used_on_last_tick,
            'remained_cash': self._total_cash,
            'initial_cash_position': self._initial_cash_position
        }

    def _compute_total_cash_ratio(self) -> float:
        return self._total_cash / self._initial_cash_position

    def _compute_used_time_ratio(self, t_tick: Optional[int] = None) -> float:
        """
        Args:
            t_tick: The tick relative to the ratio will be computed. If `None` self.t_tick will be used.

        Returns:
            Computes the ratio from t_tick to the end of the env. If t_tick is at the very beginning of the month
            the function will return 0, otherwise if it is the last day of the env it will return 1.
        """

        if t_tick is None:
            t_tick = self.t_tick
        mapped_t_tick = t_tick - self.start_tick
        total_len = self.end_tick - self.start_tick

        ratio = mapped_t_tick / total_len
        assert 0 <= ratio <= 1, 't_tick / t_datetime it is not within the current month'

        return ratio

    def _on_done(self, report: Optional[dict] = None) -> dict:
        return {
            'cash_used_on_last_tick': self.cash_used_on_last_tick
        }

    def _is_done(self) -> bool:
        return not self.has_cash()

    def _compute_render_all_graph_title(self, episode_metrics: dict) -> str:
        pa = round(episode_metrics['PA'], 4)

        return f'PA={pa}'


class ExportTeacherActionsOrderExecutionEnvironment(OrderExecutionEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.actions_store = pd.HDFStore(
            path=os.path.join(self.dataset.market.storage_dir, 'teacher_actions.h5'),
            mode='w'
        )
        self.progress_bar = tqdm(
            desc='Exported datasets',
            total=self.dataset.num_datasets
        )

    def close(self):
        super().close()

        self.progress_bar.close()
        self.actions_store.close()

    def _on_done(self, report: Optional[dict] = None) -> dict:
        results = super()._on_done(report)

        if report is not None:
            self.progress_bar.update()
            self.persist_actions(report)
        else:
            print(f'Could not persist actions for: {self.dataset}')

        return results

    def persist_actions(self, report: dict):
        key = self.create_key(self.dataset)
        actions = report['unadjusted_actions']
        action_indices = self.action_schema.get_action(actions)
        df = pd.DataFrame(
            index=report['unadjusted_dates'],
            columns=self.dataset.asset_tickers,
            data=action_indices
        )

        if key in self.actions_store:
            self.actions_store[key] = self.actions_store[key].combine_first(df)
        else:
            self.actions_store[key] = df

    @staticmethod
    def create_key(dataset: 'MultiAssetDataset') -> str:
        return '-'.join(dataset.asset_tickers)


class StudentOrderExecutionEnvironment(OrderExecutionEnvironment):
    def _create_info(self, info: dict) -> dict:
        info = super()._create_info(info)
        if self._s_t and 'teacher_action' in self._s_t:
            info['teacher_action'] = self._s_t['teacher_action']
        else:
            info['teacher_action'] = None

        return info
