import logging
import os
from abc import ABC, abstractmethod
from typing import List

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from yacht.config import TrainConfig
from yacht.data.datasets import TradingDataset, build_dataset_wrapper
from yacht.data.k_fold import build_k_fold, PurgedKFold
from yacht.environments import TradingEnv
from yacht.environments.callbacks import RewardsRenderCallback

logger = logging.getLogger(__file__)


class Trainer(ABC):
    def __init__(
            self,
            train_config: TrainConfig,
            name: str,
            agent: BaseAlgorithm,
            dataset: TradingDataset,
            train_env: TradingEnv,
    ):
        self.train_config = train_config
        self.name = name
        self.agent = agent
        self.dataset = dataset
        self.train_env = train_env

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass

    @abstractmethod
    def train(self) -> BaseAlgorithm:
        pass

    def build_callbacks(self) -> List[BaseCallback]:
        train_rewards_render_callback = RewardsRenderCallback(
            storage_path=os.path.join(self.dataset.storage_dir, 'train_rewards.png'),
            verbose=0
        )

        return [train_rewards_render_callback]


class NoEvalTrainer(Trainer):
    def train(self) -> BaseAlgorithm:
        logger.info(f'Starting training for {self.train_config.episodes} episodes.')

        logger.info(f'Train split length: {len(self.dataset)}')
        logger.info(f'Collecting steps per episode: {self.train_config.collecting_n_steps}')
        logger.info(f'Validation split length: 0\n')

        train_num_time_steps = self.train_config.episodes * self.train_config.collecting_n_steps
        log_frequency = self.train_config.log_frequency * self.train_config.collecting_n_steps
        self.agent = self.agent.learn(
            total_timesteps=train_num_time_steps,
            callback=self.build_callbacks(),
            tb_log_name=self.name,
            log_interval=log_frequency,
            n_eval_episodes=1,
            # eval_log_path=self.dataset.storage_dir,
            reset_num_timesteps=True
        )

        return self.agent


class KFoldTrainer(Trainer):
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

        super().__init__(
            train_config=train_config,
            name=name,
            agent=agent,
            dataset=dataset,
            train_env=train_env
        )

        self.val_env = val_env
        self.k_fold = k_fold

    def train(self) -> BaseAlgorithm:
        # TODO: Save logs.

        logger.info(
            f'Starting training for {self.train_config.episodes} '
            f'episodes with a k_fold {self.k_fold.n_splits} splits.'
        )
        progress_bar = tqdm(total=self.train_config.episodes)
        print()
        for k, (train_indices, val_indices) in enumerate(self.k_fold.split(X=self.dataset.get_k_folding_values())):
            logger.info(f'Train split length: {len(train_indices)}')
            logger.info(f'Collecting steps per episode: {self.train_config.collecting_n_steps}')
            logger.info(f'Validation split length: {len(val_indices)}\n')

            self.k_fold.render(self.dataset.storage_dir)

            train_dataset = build_dataset_wrapper(self.dataset, indices=train_indices)
            val_dataset = build_dataset_wrapper(self.dataset, indices=val_indices)
            self.train_env.set_dataset(train_dataset)
            self.val_env.set_dataset(val_dataset)

            # For stable baselines3 `freq` is relative to the episode time steps.
            k_fold_num_episodes = self.train_config.episodes // self.k_fold.n_splits
            k_fold_split_time_steps = k_fold_num_episodes * self.train_config.collecting_n_steps
            steps_eval_frequency = self.train_config.eval_frequency * self.train_config.collecting_n_steps
            log_frequency = self.train_config.log_frequency * self.train_config.collecting_n_steps
            self.agent = self.agent.learn(
                total_timesteps=k_fold_split_time_steps,
                callback=self.build_callbacks(),
                tb_log_name=self.name,
                log_interval=log_frequency,
                eval_env=self.val_env,
                eval_freq=steps_eval_frequency,
                n_eval_episodes=1,
                # eval_log_path=self.dataset.storage_dir,
                reset_num_timesteps=True
            )

            progress_bar.update(n=self.train_config.episodes // self.k_fold.n_splits)
            print()

        progress_bar.close()

        return self.agent

    def close(self):
        self.k_fold.close()


#######################################################################################################################

trainer_registry = {
    'NoEvalTrainer': NoEvalTrainer,
    'KFoldTrainer': KFoldTrainer
}


def build_trainer(
        config,
        agent: BaseAlgorithm,
        dataset: TradingDataset,
        train_env: TradingEnv,
        val_env: TradingEnv
) -> Trainer:
    trainer_class = trainer_registry[config.train.trainer_name]
    trainer_kwargs = {
        'train_config': config.train,
        'agent': agent,
        'name': f'{agent.__class__.__name__}_{config.environment.name}',
        'dataset': dataset,
        'train_env': train_env
    }
    if trainer_class == KFoldTrainer:
        k_fold = build_k_fold(config)

        trainer_kwargs.update({
            'val_env': val_env,
            'k_fold': k_fold
        })

    return trainer_class(**trainer_kwargs)
