import logging

from stable_baselines3.common.base_class import BaseAlgorithm

from yacht.data.datasets import TradingDataset, build_dataset_wrapper
from yacht.data.k_fold import build_k_fold
from yacht.environments import TradingEnv

logger = logging.getLogger(__file__)


class Trainer:
    def __init__(
            self,
            episodes: int,
            agent: BaseAlgorithm,
            dataset: TradingDataset,
            train_env: TradingEnv,
            val_env: TradingEnv,
            name: str,
            k_fold_splits: int
    ):
        self.episodes = episodes
        self.agent = agent
        self.dataset = dataset
        self.train_env = train_env
        self.val_env = val_env
        self.name = name
        self.k_fold_splits = k_fold_splits

    def train(self, config):
        logger.info(f'Starting training for {self.episodes} episodes.')
        for i in range(self.episodes // self.k_fold_splits):
            k_fold = build_k_fold(config.input, config.train)
            for k, (train_indices, val_indices) in enumerate(k_fold.split(X=self.dataset.get_folding_values())):
                episode = i * config.train.k_fold_splits + k
                logger.info(f'Episode - {episode}')

                self.train_on_episode(train_indices, val_indices)

    def train_on_episode(self, train_indices, val_indices):
        # TODO: Is it ok to slide the window between 2 splits of train intervals ? Data not contiguous
        train_dataset = build_dataset_wrapper(self.dataset, indices=train_indices)
        val_dataset = build_dataset_wrapper(self.dataset, indices=val_indices)

        self.train_env.set_dataset(train_dataset)
        self.val_env.set_dataset(val_dataset)

        self.agent = self.agent.learn(
            total_timesteps=1,
            callback=None,
            tb_log_name=self.name,
            log_interval=1,
            eval_env=self.val_env,
            eval_freq=len(train_indices),
            n_eval_episodes=1,
            eval_log_path=None,
            reset_num_timesteps=True
        )


#######################################################################################################################


def build_trainer(
        config,
        agent: BaseAlgorithm,
        dataset: TradingDataset,
        train_env: TradingEnv,
        val_env: TradingEnv
) -> Trainer:
    return Trainer(
        episodes=config.train.episodes,
        agent=agent,
        dataset=dataset,
        train_env=train_env,
        val_env=val_env,
        name=config.input.env,
        k_fold_splits=config.train.k_fold_splits
    )
