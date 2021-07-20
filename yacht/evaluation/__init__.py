import pprint

import logging

import pyfolio

from .backtest import *

from stable_baselines3.common.base_class import BaseAlgorithm

from yacht.agents.predict import run_agent
from yacht.environments import TradingEnv

logger = logging.getLogger(__file__)


def backtest(
        env: TradingEnv,
        agent: BaseAlgorithm,
        render: bool = True,
        render_all: bool = False,
        name: str = 'backtest',
        verbose: bool = True,
        plot: bool = False
):
    report = run_agent(
        env=env,
        agent=agent,
        render=render,
        render_all=render_all,
        name=name
    )
    backtest_results = get_daily_return(report, value_col_name='total')

    baseline_report = env.create_baseline_report()
    baseline_results = get_daily_return(baseline_report, value_col_name='total')

    backtest_statistics = timeseries.perf_stats(
        returns=backtest_results,
        factor_returns=baseline_results,
    )

    if verbose is True:
        logger.info(f'Backtest statistics [{name}] in report to the baseline: ')
        logger.info(pprint.pformat(backtest_statistics, indent=4))

    if plot:
        # This function works only in a jupyter notebook.
        with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(
                returns=backtest_results,
                benchmark_rets=baseline_results,
                set_context=False
            )
