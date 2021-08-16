from collections import defaultdict
from typing import Union, Dict

import pandas as pd
import numpy as np
from pyfolio import timeseries

from yacht import utils


def compute_backtest_metrics(report: dict, total_assets_col_name='total'):
    daily_return_report = get_daily_return(report[total_assets_col_name])

    # TODO: Add the rest of the arguments for more statistics.
    strategy_metrics = timeseries.perf_stats(
        returns=daily_return_report,
        factor_returns=None,
        positions=None,
        transactions=None,
        turnover_denom='AGB',
    )
    strategy_metrics = dict(
        zip(strategy_metrics.index, strategy_metrics.values)
    )

    # Map all indices from plain english title to snake case for consistency.
    snake_case_strategy_metrics = dict()
    for k, v in strategy_metrics.items():
        snake_case_strategy_metrics[utils.english_title_to_snake_case(k)] = v
    strategy_metrics = snake_case_strategy_metrics

    # TODO: Adapt metrics for multiple actions / prices
    price_advantage_metrics = compute_price_advantage(report)
    strategy_metrics.update(price_advantage_metrics)

    longs_shorts_ratio = compute_longs_shorts_ratio(report)
    strategy_metrics['LSR'] = longs_shorts_ratio

    final_report = pd.concat([report, daily_return_report], axis=1)

    return strategy_metrics, final_report


def get_daily_return(total_assets_over_time: Union[np.ndarray, pd.Series]) -> pd.Series:
    if isinstance(total_assets_over_time, np.ndarray):
        total_assets_over_time = pd.Series(data=total_assets_over_time)
    total_assets_over_time[total_assets_over_time == 0.] = 1e-2

    return total_assets_over_time.pct_change(1)


def compute_price_advantage(report: dict) -> dict:
    actions = report['action']
    prices = report['price']
    mean_price = np.mean(prices, axis=0)

    statistics: Dict[str, Union[list, np.ndarray]] = defaultdict(list)
    for asset_idx in range(actions.shape[1]):
        # Ignore hold actions ( action = 0) because their are irrelevant in this metric.
        positive_positions_mask = actions[:, asset_idx] > 0
        negative_positions_mask = actions[:, asset_idx] < 0
        buy_actions = actions[positive_positions_mask, asset_idx]
        sell_actions = actions[negative_positions_mask, asset_idx]

        if positive_positions_mask.any():
            statistics['buy_pa'].append(_compute_price_advantage(
                buy_actions,
                prices[positive_positions_mask, asset_idx],
                mean_price=mean_price[asset_idx],
                buy=True
            ))
            statistics['buy_weights'].append(buy_actions.sum())

        if negative_positions_mask.any():
            statistics['sell_pa'].append(_compute_price_advantage(
                sell_actions,
                prices[negative_positions_mask, asset_idx],
                mean_price=mean_price[asset_idx],
                buy=False
            ))
            statistics['sell_weights'].append(buy_actions.sum())

    statistics['buy_weights'] = np.array(statistics['buy_weights'], dtype=np.float32)
    statistics['buy_weights'] /= statistics['buy_weights'].sum()
    statistics['sell_weights'] = np.array(statistics['sell_weights'], dtype=np.float32)
    statistics['sell_weights'] /= statistics['sell_weights'].sum()

    statistics['buy_pa'] = np.array(statistics['buy_pa'], dtype=np.float32) * statistics['buy_weights']
    statistics['buy_pa'] = statistics['buy_pa'].sum()
    statistics['sell_pa'] = np.array(statistics['sell_pa'], dtype=np.float32) * statistics['sell_weights']
    statistics['sell_pa'] = statistics['sell_pa'].sum()

    return statistics


def _compute_price_advantage(
        actions: np.ndarray,
        prices: np.ndarray,
        mean_price: np.ndarray,
        buy: bool = True
) -> float:
    average_execution_price = (actions * prices).sum() / actions.sum()

    # If you buy, you want a lower AEP, else if you sell, you want a higher AEP.
    if buy:
        pa = 1 - average_execution_price / mean_price
    else:
        pa = average_execution_price / mean_price - 1

    pa *= 1e4

    return pa


def compute_longs_shorts_ratio(report: dict) -> float:
    final_num_longs = report['longs'].values[-1]
    final_num_shorts = report['short'].values[-1]

    longs_shorts_ratio = final_num_longs / (final_num_shorts + 1e-17)

    return longs_shorts_ratio.item()
