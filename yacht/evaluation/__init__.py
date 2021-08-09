import pprint

import logging
from collections import defaultdict

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from .metrics import *

from stable_baselines3.common.base_class import BaseAlgorithm

from .. import Mode, utils
from ..data.renderers import RewardsRenderer

logger = logging.getLogger(__file__)


def backtest(
        env: VecEnv,
        agent: BaseAlgorithm,
        storage_dir: str,
        mode: Mode,
        deterministic: bool = False,
        name: str = 'backtest',
        verbose: bool = True,
        plot: bool = False
):
    # Run the agent with the given policy.
    evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=env.num_envs,  # One episode for every environment.
        deterministic=deterministic,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=False,
        warn=False
    )

    # Render backtest rewards.
    total_timesteps = sum([buf_info['episode']['l'] for buf_info in env.buf_infos])
    renderer = RewardsRenderer(
        total_timesteps=total_timesteps,
        storage_dir=storage_dir,
        mode=mode
    )
    renderer.render()
    renderer.save(utils.build_rewards_path(storage_dir, mode))

    assert np.all(env.buf_dones)

    # Compute backtest statistics.
    statistics = defaultdict(list)
    for buf_info in env.buf_infos:
        backtest_metrics = buf_info['episode_metrics']

        # baseline_report = env.create_baseline_report()
        # baseline_results = get_daily_return(baseline_report, value_col_name='total')

        # backtest_statistics = timeseries.perf_stats(
        #     returns=daily_returns,
        #     # factor_returns=baseline_results,
        # )

        # if plot:
        #     # This function works only in a jupyter notebook.
        #     with pyfolio.plotting.plotting_context(font_scale=1.1):
        #         pyfolio.create_full_tear_sheet(
        #             returns=daily_returns,
        #             # benchmark_rets=baseline_results,
        #             set_context=False
        #         )

        for k, v in backtest_metrics.items():
            statistics[k].append(v)

    mean_statistics = dict()
    for k, v in statistics.items():
        if utils.is_number(v[0]):
            mean_statistics[f'{k}_mean_std'] = (np.mean(v), np.std(v))
    statistics.update(mean_statistics)

    if verbose is True:
        logger.info(f'Backtest metrics [{name}]: ')
        logger.info(pprint.pformat(statistics, indent=4))

    return statistics
