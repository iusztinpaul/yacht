import logging
from abc import ABC, abstractmethod
from typing import List

import wandb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from tqdm import tqdm

from yacht import utils
from yacht.config import Config
from yacht.data.datasets import TradingDataset, build_dataset_wrapper
from yacht.data.k_fold import build_k_fold, PurgedKFold
from yacht.environments import TradingEnv
from yacht.environments.callbacks import LoggerCallback, WandBCallback

logger = logging.getLogger(__file__)


class Trainer(ABC):
    def __init__(
            self,
            config: Config,
            name: str,
            agent: BaseAlgorithm,
            dataset: TradingDataset,
            train_env: TradingEnv,
            save: bool = True
    ):
        self.config = config
        self.train_config = config.train
        self.name = name
        self.agent = agent
        self.dataset = dataset
        self.train_env = train_env

        self.save = save

    def __enter__(self):
        self.agent.policy.train()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save is True:
            save_path = utils.build_last_checkpoint_path(self.dataset.storage_dir)
            self.agent.save(
                path=save_path
            )

        self.close()

    def close(self):
        pass

    @abstractmethod
    def train(self) -> BaseAlgorithm:
        pass

    def build_callbacks(self) -> List[BaseCallback]:
        callbacks = [
            LoggerCallback(
                total_timesteps=self.train_config.total_timesteps,
            )
        ]

        if utils.get_experiment_tracker_name(self.dataset.storage_dir) == 'wandb':
            wandb_callback = WandBCallback(storage_dir=self.dataset.storage_dir)
            callbacks.append(wandb_callback)

        return callbacks


class NoEvalTrainer(Trainer):
    def train(self) -> BaseAlgorithm:
        logger.info(f'Training for {self.train_config.total_timesteps} timesteps.')
        logger.info(f'Train split length: {len(self.dataset)}')
        logger.info(f'Validation split length: 0\n')

        self.agent = self.agent.learn(
            total_timesteps=self.train_config.total_timesteps,
            callback=self.build_callbacks(),
            tb_log_name=self.name,
            log_interval=self.train_config.collecting_n_steps,
        )

        return self.agent

    def build_callbacks(self) -> List[BaseCallback]:
        callbacks = super().build_callbacks()
        # Save the best model relative to the training environment.
        callbacks.append(
            EvalCallback(
                eval_env=self.train_env,
                eval_freq=self.train_config.collecting_n_steps * 5,
                log_path=self.dataset.storage_dir,
                best_model_save_path=utils.build_best_checkpoint_dir(self.dataset.storage_dir),
                deterministic=True,
                verbose=False
            )
        )

        return callbacks


class KFoldTrainer(Trainer):
    def __init__(
            self,
            config: Config,
            name: str,
            agent: BaseAlgorithm,
            dataset: TradingDataset,
            train_env: TradingEnv,
            val_env: TradingEnv,
            k_fold: PurgedKFold,
    ):
        train_config = config.train
        assert train_config.collect_n_times >= train_config.eval_frequency

        super().__init__(
            config=config,
            name=name,
            agent=agent,
            dataset=dataset,
            train_env=train_env
        )

        self.val_env = val_env
        self.k_fold = k_fold

    def train(self) -> BaseAlgorithm:
        logger.info(
            f'Training for {self.train_config.total_timesteps} timesteps, '
            f'with a k_fold {self.k_fold.n_splits} splits.'
        )
        progress_bar = tqdm(total=self.train_config.collect_n_times)
        print()
        for k, (train_indices, val_indices) in enumerate(self.k_fold.split(X=self.dataset.get_prices())):
            k_fold_split_timesteps = self.train_config.total_timesteps // self.k_fold.n_splits

            logger.info(f'Training for {k_fold_split_timesteps} timesteps.')
            logger.info(f'Train split length: {len(train_indices)}')
            logger.info(f'Validation split length: {len(val_indices)}\n')

            self.k_fold.render(self.dataset.storage_dir)

            train_dataset = build_dataset_wrapper(self.dataset, indices=train_indices)
            val_dataset = build_dataset_wrapper(self.dataset, indices=val_indices)
            self.train_env.set_dataset(train_dataset)
            self.val_env.set_dataset(val_dataset)

            self.agent = self.agent.learn(
                total_timesteps=k_fold_split_timesteps,
                callback=self.build_callbacks(),
                tb_log_name=self.name,
                log_interval=self.train_config.collecting_n_steps,
                eval_env=self.val_env,
                eval_freq=self.train_config.collecting_n_steps,
                n_eval_episodes=1,
                eval_log_path=self.dataset.storage_dir,
                reset_num_timesteps=True
            )

            progress_bar.update(n=self.train_config.collect_n_times // self.k_fold.n_splits)
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
        val_env: TradingEnv,
        save: bool
) -> Trainer:
    trainer_class = trainer_registry[config.train.trainer_name]
    trainer_kwargs = {
        'config': config,
        'agent': agent,
        'name': f'{agent.__class__.__name__}_{config.environment.name}',
        'dataset': dataset,
        'train_env': train_env,
        'save': save
    }
    if trainer_class == KFoldTrainer:
        k_fold = build_k_fold(config)

        trainer_kwargs.update({
            'val_env': val_env,
            'k_fold': k_fold
        })

    return trainer_class(**trainer_kwargs)
