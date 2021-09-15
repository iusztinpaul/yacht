from abc import ABC
from typing import List, Union, Optional

import wandb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm

from yacht import utils, Mode
from yacht.agents import build_agent
from yacht.config import Config
from yacht.data.datasets import AssetDataset, build_dataset_wrapper, build_dataset, SampleAssetDataset
from yacht.data.k_fold import PurgedKFold
from yacht.environments import build_env, MetricsVecEnvWrapper
from yacht.environments.callbacks import LoggerCallback, MetricsEvalCallback, RewardsRenderCallback
from yacht.logger import Logger
from yacht.utils.wandb import WandBCallback


class Trainer(ABC):
    def __init__(
            self,
            config: Config,
            name: str,
            agent: BaseAlgorithm,
            train_dataset: SampleAssetDataset,
            validation_dataset: SampleAssetDataset,
            train_env: VecEnv,
            validation_env: MetricsVecEnvWrapper,
            logger: Logger,
            mode: Mode,
            save: bool = True
    ):
        self.config = config
        self.name = name
        self.agent = agent
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_env = train_env
        self.validation_env = validation_env
        self.logger = logger
        self.mode = mode
        self.save = save

        assert self.train_dataset.storage_dir == self.validation_dataset.storage_dir
        self.storage_dir = self.train_dataset.storage_dir

        if self.mode.is_fine_tuning():
            self.total_timesteps = self.config.train.fine_tune_total_timesteps
        else:
            self.total_timesteps = self.config.train.total_timesteps

        self.agent.policy.train()

    def close(self):
        self.agent.policy.eval()

        if self.save is True:
            save_path = utils.build_last_checkpoint_path(self.storage_dir, self.mode)
            self.agent.save(
                path=save_path
            )

            # TODO: Find a way to add this line to the wandb classes for consistency.
            if utils.get_experiment_tracker_name(self.storage_dir) == 'wandb':
                wandb.save(save_path)

        self.train_env.close()
        self.validation_env.close()
        self.train_dataset.close()
        self.validation_dataset.close()

    def train(self) -> BaseAlgorithm:
        self.before_train_log()
        self.agent = self.agent.learn(
            total_timesteps=self.total_timesteps,
            callback=self.build_callbacks(),
            log_interval=self.config.train.collecting_n_steps,
        )
        self.after_train_log()

        return self.agent

    def before_train_log(self):
        self.logger.info(f'Started training with {self.__class__.__name__} in mode: {self.mode.value}.')
        self.logger.info(f'Training for {self.config.train.total_timesteps} timesteps.')
        self.logger.info(f'Train split length: {self.train_dataset.num_sampling_period}')
        self.logger.info(f'Validation split length: {self.validation_dataset.num_sampling_period}.\n')

    def after_train_log(self):
        self.logger.info(f'Training finished for {self.__class__.__name__} in mode: {self.mode.value}.')

    def build_callbacks(self) -> List[BaseCallback]:
        callbacks = [
            LoggerCallback(
                logger=self.logger,
                log_frequency=self.config.meta.log_frequency_steps,
                total_timesteps=self.total_timesteps
            ),
            MetricsEvalCallback(
                eval_env=self.validation_env,
                metrics_to_save_best_on=list(self.config.meta.metrics_to_save_best_on),
                mode=Mode.BacktestValidation,
                n_eval_episodes=len(self.validation_dataset.datasets),
                eval_freq=self.config.train.collecting_n_steps * 2,
                log_path=utils.build_log_dir(self.storage_dir),
                best_model_save_path=utils.build_best_checkpoint_dir(self.storage_dir, self.mode),
                deterministic=self.config.input.backtest.deterministic,
                verbose=True
            ),
            # RewardsRenderCallback(
            #     total_timesteps=self.total_timesteps,
            #     storage_dir=self.storage_dir,
            #     mode=self.mode
            # )
        ]

        if utils.get_experiment_tracker_name(self.storage_dir) == 'wandb':
            wandb_callback = WandBCallback(storage_dir=self.storage_dir, mode=self.mode)
            callbacks.append(wandb_callback)

        return callbacks


class KFoldTrainer(Trainer):
    def __init__(
            self,
            config: Config,
            name: str,
            agent: BaseAlgorithm,
            dataset: AssetDataset,
            train_env: VecEnv,
            validation_env: VecEnv,
            logger: Logger,
            k_fold: PurgedKFold,
            save: bool = True,
    ):

        super().__init__(
            config=config,
            name=name,
            agent=agent,
            dataset=dataset,
            train_env=train_env,
            validation_env=validation_env,
            logger=logger,
            save=save
        )

        self.k_fold = k_fold

    def close(self):
        self.k_fold.close()

    def train(self) -> BaseAlgorithm:
        self.logger.info(
            f'Training for {self.config.train.total_timesteps} timesteps, '
            f'with a k_fold {self.k_fold.n_splits} splits.'
        )
        progress_bar = tqdm(total=self.config.train.collect_n_times)
        print()
        for k, (train_indices, val_indices) in enumerate(self.k_fold.split(X=self.train_dataset.get_prices())):
            k_fold_split_timesteps = self.config.train.total_timesteps // self.k_fold.n_splits

            self.logger.info(f'Training for {k_fold_split_timesteps} timesteps.')
            self.logger.info(f'Train split length: {len(train_indices)}')
            self.logger.info(f'Validation split length: {len(val_indices)}\n')

            self.k_fold.render(self.storage_dir)

            train_dataset = build_dataset_wrapper(self.train_dataset, indices=train_indices)
            val_dataset = build_dataset_wrapper(self.train_dataset, indices=val_indices)
            self.train_env.set_dataset(train_dataset)
            self.validation_env.set_dataset(val_dataset)

            self.agent = self.agent.learn(
                total_timesteps=k_fold_split_timesteps,
                callback=self.build_callbacks(),
                tb_log_name=self.name,
                log_interval=self.config.train.collecting_n_steps,
                eval_env=self.validation_env,
                eval_freq=self.config.train.collecting_n_steps,
                n_eval_episodes=1,
                eval_log_path=self.storage_dir,
                reset_num_timesteps=True
            )

            progress_bar.update(n=self.config.train.collect_n_times // self.k_fold.n_splits)
            print()

        progress_bar.close()

        return self.agent


#######################################################################################################################

trainer_registry = {
    'Trainer': Trainer,
    'KFoldTrainer': KFoldTrainer
}


def run_train(config: Config, logger: Logger, storage_dir: str, resume_training: bool):
    trainer = build_trainer(
        config=config,
        storage_dir=storage_dir,
        resume_training=resume_training,
        mode=Mode.Train,
        logger=logger,
        save=True
    )
    agent = trainer.train()
    trainer.close()

    # Fine tune agent.
    trainer = build_trainer(
        config=config,
        storage_dir=storage_dir,
        resume_training=False,
        mode=Mode.FineTuneTrain,
        logger=logger,
        save=True,
        agent=agent
    )
    trainer.train()
    trainer.close()


def build_trainer(
        config,
        storage_dir: str,
        resume_training: bool,
        mode: Mode,
        logger: Logger,
        save: bool,
        agent: Optional[BaseAlgorithm] = None
) -> Trainer:
    assert mode.is_trainable()

    train_dataset = build_dataset(config, logger, storage_dir, mode=mode)
    train_env = build_env(config, train_dataset, logger, mode=mode)
    validation_dataset = build_dataset(config, logger, storage_dir, mode=Mode.BacktestValidation)
    validation_env = build_env(config, validation_dataset, logger, mode=Mode.BacktestValidation)

    if agent is None:
        agent = build_agent(
            config=config,
            env=train_env,
            logger=logger,
            storage_dir=storage_dir,
            resume=resume_training,
            agent_from='latest'
        )
    else:
        assert mode.is_fine_tuning()

        agent.set_env(train_env)

    trainer_class = trainer_registry[config.train.trainer_name]
    trainer_kwargs = {
        'config': config,
        'agent': agent,
        'name': utils.create_project_name(config, train_dataset.storage_dir),
        'train_dataset': train_dataset,
        'validation_dataset': validation_dataset,
        'train_env': train_env,
        'validation_env': validation_env,
        'logger': logger,
        'mode': mode,
        'save': save
    }

    return trainer_class(**trainer_kwargs)
