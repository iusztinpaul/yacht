import pprint

import logging

import numpy as np
import pyfolio
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from .backtest import *

from stable_baselines3.common.base_class import BaseAlgorithm

from ..utils.sequence import get_daily_return

logger = logging.getLogger(__file__)


def backtest(
        env: VecEnv,
        agent: BaseAlgorithm,
        deterministic: bool = False,
        name: str = 'backtest',
        verbose: bool = True,
        plot: bool = False
):
    evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=env.num_envs,
        deterministic=deterministic,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=False,
        warn=False
    )

    assert np.all(env.buf_dones)

    statistics = dict()
    for buf_info in env.buf_infos:
        report = buf_info['report']
        backtest_results = get_daily_return(report, value_col_name='total')

        # baseline_report = env.create_baseline_report()
        # baseline_results = get_daily_return(baseline_report, value_col_name='total')

        backtest_statistics = timeseries.perf_stats(
            returns=backtest_results,
            # factor_returns=baseline_results,
        )

        if verbose is True:
            logger.info(f'Backtest statistics [{buf_info["ticker"]} - {name}]: ')
            logger.info(pprint.pformat(backtest_statistics, indent=4))

        if plot:
            # This function works only in a jupyter notebook.
            with pyfolio.plotting.plotting_context(font_scale=1.1):
                pyfolio.create_full_tear_sheet(
                    returns=backtest_results,
                    # benchmark_rets=baseline_results,
                    set_context=False
                )

        statistics[buf_info['ticker']] = backtest_statistics

    return statistics
