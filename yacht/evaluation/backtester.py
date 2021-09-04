import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from yacht import Mode, utils
from yacht.agents import build_agent
from yacht.config import Config
from yacht.data.datasets import build_dataset, SampleAssetDataset
from yacht.data.renderers import RewardsRenderer
from yacht.environments import build_env, MetricsVecEnvWrapper
from yacht.logger import Logger


class BackTester:
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
            n_eval_episodes=self.env.num_envs,  # One episode for every environment.
            deterministic=self.config.input.backtest.deterministic,
            render=False,
            callback=None,
            reward_threshold=None,
            return_episode_rewards=False,
            warn=False
        )

        # Render backtest rewards.
        total_timesteps = sum([buf_info['episode']['l'] for buf_info in self.env.buf_infos])
        renderer = RewardsRenderer(
            total_timesteps=total_timesteps,
            storage_dir=self.dataset.storage_dir,
            mode=self.mode
        )
        renderer.render()
        renderer.save(utils.build_rewards_path(self.dataset.storage_dir, self.mode))

        assert np.all(self.env.buf_dones), 'Cannot compute metrics on undone environments.'

        return self.env.mean_metrics, self.env.std_metrics

    def close(self):
        self.env.close()
        self.dataset.close()


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
