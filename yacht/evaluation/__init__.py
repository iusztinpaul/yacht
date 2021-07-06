import logging

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
        verbose: bool = True
):
    report = run_agent(
        env=env,
        agent=agent,
        render=render,
        render_all=render_all,
        name=name
    )

    backtest_results = compute_backtest_results(
        report,
        value_col_name='Total Value',
    )

    if verbose:
        logger.info(f'Backtest results [{name}]:')
        logger.info(backtest_results)

    return backtest_results
