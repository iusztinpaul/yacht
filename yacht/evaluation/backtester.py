from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, List

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from yacht import Mode
from yacht.agents import build_agent
from yacht.config import Config
from yacht.data.datasets import build_dataset, SampleAssetDataset
from yacht.environments import build_env, MetricsVecEnvWrapper
from yacht.logger import Logger


class BackTester(ABC):
    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def close(self):
        pass


class SimpleBackTester(BackTester):
    def __init__(
            self,
            config: Config,
            dataset: SampleAssetDataset,
            env: MetricsVecEnvWrapper,
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
        # Run the agent with the given policy.
        evaluate_policy(
            model=self.agent,
            env=self.env,
            n_eval_episodes=len(self.dataset.datasets),  # Evaluate all possible datasets.
            deterministic=self.config.input.backtest.deterministic,
            render=False,
            callback=None,
            reward_threshold=None,
            return_episode_rewards=False,
            warn=False
        )

        assert np.all(self.env.buf_dones), 'Cannot compute metrics on undone environments.'

        return self.env.mean_metrics, self.env.std_metrics

    def close(self):
        self.env.close()
        self.dataset.close()


class ListBackTester(BackTester):
    def __init__(
            self,
            backtesters: List[BackTester]
    ):
        self.backtesters = backtesters

    def test(self):
        return [b.test() for b in self.backtesters]

    def close(self):
        for b in self.backtesters:
            b.close()


#######################################################################################################################


def build_backtester(
        config: Config,
        logger: Logger,
        storage_dir: str,
        mode: Mode,
        agent_from: str,
        market_storage_dir: Optional[str]
) -> Optional[BackTester]:
    if mode.is_best_metric():
        if 'reward' not in config.meta.metrics_to_load_best_on:
            config.meta.metrics_to_load_best_on.append('reward')
    else:
        assert len(config.meta.metrics_to_load_best_on) <= 1, 'Cannot load from multiple metrics in the current setup.'

    backtesters = []
    for metric in config.meta.metrics_to_load_best_on:
        dataset = build_dataset(config, logger, storage_dir, mode=mode, market_storage_dir=market_storage_dir)
        if dataset is None:
            logger.info('Could not create the dataset.')
            return None

        env = build_env(config, dataset, logger, mode=mode, load_best_metric=metric)
        agent = build_agent(
            config,
            env,
            logger=logger,
            storage_dir=storage_dir,
            resume=True,
            agent_from=agent_from,
            best_metric=metric
        )

        backtesters.append(
            SimpleBackTester(
                config=config,
                dataset=dataset,
                env=env,
                agent=agent,
                logger=logger,
                mode=mode,
            )
        )

    return ListBackTester(
        backtesters=backtesters
    )
