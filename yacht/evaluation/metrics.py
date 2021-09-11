from collections import defaultdict
from typing import Union, Dict, Tuple, List

import pandas as pd
import numpy as np
from pyfolio import timeseries

from yacht import utils


def compute_backtest_metrics(
        report: Dict[str, Union[list, np.ndarray]],
        total_assets_col_name='total_assets',
        buy: bool = True
) -> Tuple[dict, dict]:
    daily_returns = get_daily_return(report[total_assets_col_name])
    report['daily_returns'] = daily_returns.values

    # TODO: Add the rest of the arguments for more statistics.
    strategy_metrics = timeseries.perf_stats(
        returns=daily_returns,
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

    strategy_metrics['PA'] = aggregate_price_advantage(report, buy=buy)
    strategy_metrics['LSR'] = compute_longs_shorts_ratio(report)

    return strategy_metrics, report


def get_daily_return(total_assets_over_time: Union[np.ndarray, pd.Series]) -> pd.Series:
    if isinstance(total_assets_over_time, np.ndarray):
        total_assets_over_time = pd.Series(data=total_assets_over_time)
    total_assets_over_time[total_assets_over_time == 0.] = 1e-2

    return total_assets_over_time.pct_change(1)


def aggregate_price_advantage(report: dict, buy: bool) -> np.ndarray:
    actions = report['action']
    prices = report['price']
    mean_price = np.mean(prices, axis=0)

    statistics: Dict[str, Union[list, np.ndarray]] = defaultdict(list)
    for asset_idx in range(actions.shape[1]):
        if buy:
            # Accept 0 actions for the case when all actions are 0. It will not influence the metric.
            positive_positions_mask = actions[:, asset_idx] >= 0
            buy_actions = actions[positive_positions_mask, asset_idx]

            if positive_positions_mask.any():
                statistics['PA'].append(compute_price_advantage(
                    buy_actions,
                    prices[positive_positions_mask, asset_idx],
                    mean_price=mean_price[asset_idx],
                    buy=True
                ))
                statistics['weights'].append(buy_actions.sum())
            else:
                raise RuntimeError('No buy actions to compute PA metric.')
        else:
            # Accept 0 actions for the case when all actions are 0. It will not influence the metric.
            negative_positions_mask = actions[:, asset_idx] <= 0
            sell_actions = actions[negative_positions_mask, asset_idx]

            if negative_positions_mask.any():
                statistics['PA'].append(compute_price_advantage(
                    sell_actions,
                    prices[negative_positions_mask, asset_idx],
                    mean_price=mean_price[asset_idx],
                    buy=False
                ))
                statistics['weights'].append(sell_actions.sum())
            else:
                raise RuntimeError('No sell actions to compute PA metric.')

    statistics['weights'] = np.array(statistics['weights'], dtype=np.float32)
    statistics['weights'] /= statistics['weights'].sum()
    statistics['PA'] = np.array(statistics['PA'], dtype=np.float32) * statistics['weights']
    statistics['PA'] = statistics['PA'].sum()

    return statistics['PA']


def compute_price_advantage(
        actions: np.ndarray,
        prices: np.ndarray,
        mean_price: np.ndarray,
        buy: bool = True
) -> float:
    try:
        average_execution_price = (actions * prices).sum() / actions.sum()
    except ZeroDivisionError:
        return 0

    # If you buy, you want a lower AEP, else if you sell, you want a higher AEP.
    if buy:
        pa = 1 - average_execution_price / mean_price
    else:
        pa = average_execution_price / mean_price - 1

    pa *= 1e4

    return pa


def compute_glr_ratio(pa_values: Union[List[float], np.ndarray]) -> np.ndarray:
    if isinstance(pa_values, list):
        pa_values = np.array(pa_values, dtype=np.float32)

    positive_pa_values = pa_values[pa_values >= 0]
    negative_pa_values = pa_values[pa_values < 0]

    return positive_pa_values.mean() / np.abs(negative_pa_values.mean())


def compute_longs_shorts_ratio(report: dict) -> float:
    final_num_longs = report['longs'][-1]
    final_num_shorts = report['shorts'][-1]

    longs_shorts_ratio = final_num_longs / (final_num_shorts + 1e-17)

    return longs_shorts_ratio
