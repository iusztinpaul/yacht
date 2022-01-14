from abc import ABC
from typing import List, Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from yacht import utils, Mode
from yacht.agents import build_agent
from yacht.config import Config
from yacht.data.datasets import build_dataset, SampleAssetDataset
from yacht.environments import build_env, MetricsVecEnvWrapper
from yacht.environments.callbacks import LoggerCallback, MetricsEvalCallback, LastCheckpointCallback
from yacht.evaluation import run_backtest
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

    def close(self):
        if self.save is True:
            save_path = utils.build_last_checkpoint_path(self.storage_dir, self.mode)
            self.agent.save(path=save_path)

        self.train_env.close()
        self.validation_env.close()
        self.train_dataset.close()
        self.validation_dataset.close()

    def train(self) -> BaseAlgorithm:
        self.agent.policy.train()

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
                plateau_max_n_steps=self.config.meta.plateau_max_n_steps,
                mode=Mode.BacktestValidation,
                n_eval_episodes=len(self.validation_dataset.datasets),
                eval_freq=self.config.train.collecting_n_steps * self.config.environment.n_envs,
                log_path=utils.build_log_dir(self.storage_dir),
                best_model_save_path=utils.build_checkpoints_dir(self.storage_dir, self.mode),
                deterministic=self.config.input.backtest.deterministic,
                logger=self.logger,
                verbose=1
            ),
            LastCheckpointCallback(
                save_freq=self.config.train.collecting_n_steps * self.config.environment.n_envs,
                save_path=utils.build_last_checkpoint_path(self.storage_dir, self.mode),
            )
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


#######################################################################################################################

trainer_registry = {
    'Trainer': Trainer,
}


def run_train(
        config: Config,
        logger: Logger,
        storage_dir: str,
        agent_from: Optional[str] = None,
        market_storage_dir: Optional[str] = None
):
    trainer = build_trainer(
        config=config,
        storage_dir=storage_dir,
        resume=agent_from is not None,
        mode=Mode.Train,
        logger=logger,
        save=True,
        agent_from=agent_from,
        market_storage_dir=market_storage_dir,
    )
    trainer.train()
    trainer.close()
    # Run a backtest on the validation split to see the best results more explicitly for the main training.
    run_backtest(
        config=config,
        logger=logger,
        storage_dir=storage_dir,
        agent_from='best-train',
        mode=Mode.BestMetricBacktestValidation,
        market_storage_dir=market_storage_dir
    )

    if config.train.fine_tune_total_timesteps >= config.train.collecting_n_steps:
        # Fine tune agent.
        trainer = build_trainer(
            config=config,
            storage_dir=storage_dir,
            resume=True,
            mode=Mode.FineTuneTrain,
            logger=logger,
            save=True,
            agent_from='best-train',
            market_storage_dir=market_storage_dir
        )
        trainer.train()
        trainer.close()
        # Run a backtest on the validation split to see the best results more explicitly for fine-tuning.
        run_backtest(
            config=config,
            logger=logger,
            storage_dir=storage_dir,
            agent_from='best-fine-tune',
            mode=Mode.BestMetricBacktestValidation,
            market_storage_dir=market_storage_dir
        )
    else:
        logger.info(
            f'Fine tuning is stopped: '
            f'fine_tune_total_timesteps [ = {config.train.fine_tune_total_timesteps}] < '
            f'collecting_n_steps [ = {config.train.collecting_n_steps}]'
        )


def build_trainer(
        config,
        storage_dir: str,
        resume: bool,
        mode: Mode,
        logger: Logger,
        save: bool,
        agent_from: Optional[str] = None,
        agent: Optional[BaseAlgorithm] = None,
        market_storage_dir: Optional[str] = None
) -> Trainer:
    assert mode.is_trainable()
    if resume:
        assert bool(agent_from) or bool(agent)

    train_dataset = build_dataset(
        config,
        logger,
        storage_dir,
        mode=mode,
        market_storage_dir=market_storage_dir
    )
    if train_dataset is None:
        raise RuntimeError('Could not create training dataset.')

    train_env = build_env(config, train_dataset, logger, mode=mode)
    validation_dataset = build_dataset(
        config,
        logger,
        storage_dir,
        mode=Mode.BacktestValidation,
        market_storage_dir=market_storage_dir
    )
    if validation_dataset is None:
        raise RuntimeError('Could not create validation dataset.')
    validation_env = build_env(config, validation_dataset, logger, mode=Mode.BacktestValidation)

    if agent is None:
        best_metrics_to_load = config.meta.metrics_to_load_best_on
        if len(best_metrics_to_load) > 1:
            best_metric = best_metrics_to_load[0]
        else:
            best_metric = 'reward'
        if resume:
            logger.info(f'Loading agent from best metric: {best_metric}')

        agent = build_agent(
            config=config,
            env=train_env,
            logger=logger,
            storage_dir=storage_dir,
            resume=resume,
            agent_from=agent_from,
            best_metric=best_metric
        )
    else:
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
