import pprint
from collections import defaultdict
from typing import List

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from yacht import Mode, utils
from yacht.agents import build_agent
from yacht.config import Config
from yacht.data.datasets import build_dataset, ChooseAssetDataset
from yacht.data.renderers import RewardsRenderer
from yacht.environments import build_env
from yacht.logger import Logger


class BackTester:
    def __init__(
            self,
            config: Config,
            dataset: ChooseAssetDataset,
            env: VecEnv,
            agent: BaseAlgorithm,
            logger: Logger,
            mode: Mode,
            name: str = None
    ):
        self.config = config
        self.dataset = dataset
        self.env = env
        self.agent = agent
        self.logger = logger
        self.mode = mode
        self.name = name if name is not None else self.mode.value

    def test(self):
        self.backtest(
            self.env,
            self.agent,
            storage_dir=self.dataset.storage_dir,
            mode=self.mode,
            deterministic=self.config.input.backtest.deterministic,
            name=self.name
        )

    def close(self):
        self.env.close()
        self.dataset.close()

    def backtest(
            self,
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

        assert np.all(env.buf_dones), 'Cannot compute metrics on undone environments.'

        # Compute mean & std on all the experiments.
        statistics = self.aggregate_metrics(infos=env.buf_infos)
        if verbose is True:
            self.logger.info(f'Backtest metrics [{name}]: ')
            self.logger.info(pprint.pformat(statistics, indent=4))

        return statistics

    @classmethod
    def aggregate_metrics(cls, infos: List[dict]) -> dict:
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


#######################################################################################################################


def build_backtester(config: Config, logger: Logger, storage_dir: str, mode: Mode, agent_from: str) -> BackTester:
    dataset = build_dataset(config, logger, storage_dir, mode=mode)
    env = build_env(config, dataset, logger, mode=mode)
    agent = build_agent(
        config,
        env,
        logger=logger,
        storage_dir=storage_dir,
        resume=True,
        agent_from=agent_from
    )

    return BackTester(
        config=config,
        dataset=dataset,
        env=env,
        agent=agent,
        logger=logger,
        mode=mode,
    )
