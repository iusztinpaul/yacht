import logging

from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

from yacht.config import TrainConfig
from yacht.data.datasets import TradingDataset, build_dataset_wrapper
from yacht.data.k_fold import build_k_fold, PurgedKFold
from yacht.environments import TradingEnv

logger = logging.getLogger(__file__)


class Trainer:
    def __init__(
            self,
            train_config: TrainConfig,
            name: str,
            agent: BaseAlgorithm,
            dataset: TradingDataset,
            train_env: TradingEnv,
            val_env: TradingEnv,
            k_fold: PurgedKFold,
    ):
        assert train_config.episodes >= train_config.eval_frequency

        self.train_config = train_config
        self.name = name
        self.agent = agent
        self.dataset = dataset
        self.train_env = train_env
        self.val_env = val_env
        self.k_fold = k_fold

    def train(self):
        # TODO: Save logs.

        logger.info(f'Starting training for {self.train_config.episodes} episodes.')
        progress_bar = tqdm(total=self.train_config.episodes)
        print()
        for k, (train_indices, val_indices) in enumerate(self.k_fold.split(X=self.dataset.get_k_folding_values())):
            self.k_fold.render(self.dataset.storage_dir)
            logger.info(f'\nTrain split length: {len(train_indices)}')
            logger.info(f'Collecting steps per episode: {self.train_config.collecting_n_steps}')
            logger.info(f'Validation split length: {len(val_indices)}\n')

            train_dataset = build_dataset_wrapper(self.dataset, indices=train_indices)
            val_dataset = build_dataset_wrapper(self.dataset, indices=val_indices)
            self.train_env.set_dataset(train_dataset)
            self.val_env.set_dataset(val_dataset)

            k_fold_num_episodes = self.train_config.episodes // self.k_fold.n_splits
            k_fold_split_time_steps = k_fold_num_episodes * self.train_config.collecting_n_steps
            # For stable baselines3 `eval_freq` is relative to the episode time steps.
            steps_eval_frequency = self.train_config.eval_frequency * self.train_config.collecting_n_steps
            self.agent = self.agent.learn(
                total_timesteps=k_fold_split_time_steps,
                callback=None,
                tb_log_name=self.name,
                log_interval=self.train_config.log_frequency,
                eval_env=self.val_env,
                eval_freq=steps_eval_frequency,
                n_eval_episodes=1,
                eval_log_path=None,
                reset_num_timesteps=True
            )

            progress_bar.update(n=self.train_config.episodes // self.k_fold.n_splits)
            print()

        progress_bar.close()

        return self.agent

    def close(self):
        self.k_fold.close()


#######################################################################################################################


def build_trainer(
        config,
        agent: BaseAlgorithm,
        dataset: TradingDataset,
        train_env: TradingEnv,
        val_env: TradingEnv
) -> Trainer:
    k_fold = build_k_fold(config)

    return Trainer(
        train_config=config.train,
        agent=agent,
        name=config.environment.name,
        dataset=dataset,
        train_env=train_env,
        val_env=val_env,
        k_fold=k_fold,
    )
