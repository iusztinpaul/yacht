import pprint

import logging
from collections import defaultdict
from typing import List

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from .metrics import *

from stable_baselines3.common.base_class import BaseAlgorithm

from .. import Mode, utils
from ..agents import build_agent
from ..config import Config
from ..data.datasets import build_dataset
from ..data.renderers import RewardsRenderer
from ..environments import build_env

logger = logging.getLogger(__file__)


# TODO: Mode this logic to a class 'Backtester'
def run_backtest(config: Config, storage_dir: str, agent_path: str):
    logger.info('Starting backtesting...')

    trainval_dataset = build_dataset(config, storage_dir, mode=Mode.BacktestTrain)
    trainval_env = build_env(config, trainval_dataset, mode=Mode.BacktestTrain)
    agent = build_agent(
        config,
        trainval_env,
        storage_dir,
        resume=True,
        agent_path=agent_path
    )
    backtest(
        trainval_env,
        agent,
        storage_dir=storage_dir,
        mode=Mode.BacktestTrain,
        deterministic=config.input.backtest.deterministic,
        name=Mode.BacktestTrain.value
    )

    test_dataset = build_dataset(config, storage_dir, mode=Mode.Backtest)
    test_env = build_env(config, test_dataset, mode=Mode.Backtest)
    agent = build_agent(
        config,
        test_env,
        storage_dir,
        resume=True,
        agent_path=agent_path
    )
    backtest(
        test_env,
        agent,
        storage_dir=storage_dir,
        mode=Mode.Backtest,
        deterministic=config.input.backtest.deterministic,
        name=Mode.Backtest.value
    )

    trainval_dataset.close()
    trainval_env.close()
    test_dataset.close()
    test_env.close()


def backtest(
        env: VecEnv,
        agent: BaseAlgorithm,
        storage_dir: str,
        mode: Mode,
        deterministic: bool = False,
        name: str = 'backtest',
        verbose: bool = True
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

    statistics = aggregate_metrics(infos=env.buf_infos)
    if verbose is True:
        logger.info(f'Backtest metrics [{name}]: ')
        logger.info(pprint.pformat(statistics, indent=4))

    return statistics


def aggregate_metrics(infos: List[dict]) -> dict:
    # Aggregate every specific metric to a list.
    statistics = defaultdict(list)
    for buf_info in infos:
        backtest_metrics = buf_info['episode_metrics']

        for k, v in backtest_metrics.items():
            statistics[k].append(v)

    # Compute the mean & std.
    mean_statistics = dict()
    for k, v in statistics.items():
        if utils.is_number(v[0]):
            mean_statistics[f'{k}_mean_std'] = (np.mean(v), np.std(v))
    statistics.update(mean_statistics)

    return mean_statistics
