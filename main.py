import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import wandb

from yacht.config import load_config, export_config
from yacht.data.datasets import build_dataset
from yacht.environments import build_env, Mode
from yacht import utils, evaluation
from yacht import environments
from yacht.agents import build_agent
from yacht.trainer import build_trainer
from yacht.utils.wandb import WandBContext

logger = logging.getLogger(__file__)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('train', 'backtest', 'max_possible_profit'))
parser.add_argument(
    '--config_file_name',
    required=True,
    help='Name of the *.config file from the configuration dir.'
)
parser.add_argument('--save_agent', default=1, help='Save agent checkpoints or not.')
parser.add_argument('--resume_training', default=False, action='store_true', help='Resume training or not.')
parser.add_argument(
    '--agent_path',
    default=None,
    help='File path to the *.pt file that you want to resume from. If None it will resume from last_checkpoint.pt'
)
parser.add_argument('--storage_dir', required=True, help='Directory where your model & logs will be saved.')
parser.add_argument('--logger_level', default='info', choices=('info', 'debug', 'warn'))


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = parser.parse_args()
    if '.' == args.storage_dir[0]:
        storage_dir = os.path.join(ROOT_DIR, args.storage_dir[2:])
    else:
        storage_dir = args.storage_dir

    utils.load_env(root_dir=ROOT_DIR)
    wandb.login(key=os.environ['WANDB_API_KEY'])
    log_dir = utils.setup_logger(
        level=args.logger_level,
        storage_dir=storage_dir
    )
    environments.register_gym_envs()
    config = load_config(os.path.join(ROOT_DIR, 'yacht', 'config', 'configs', args.config_file_name))
    export_config(config, storage_dir)
    logger.info(f'Config:\n{config}')

    with WandBContext(config, storage_dir):
        mode = Mode.from_string(args.mode)
        if mode == Mode.Train:
            dataset = build_dataset(config, storage_dir, mode=Mode.Train)
            train_env = build_env(config, dataset, mode=Mode.Train)
            val_env = build_env(config, dataset, mode=Mode.Validation)
            agent = build_agent(
                config=config,
                env=train_env,
                storage_dir=storage_dir,
                resume=args.resume_training,
                agent_path=args.agent_path
            )

            trainer = build_trainer(
                config=config,
                agent=agent,
                dataset=dataset,
                train_env=train_env,
                val_env=val_env,
                save=bool(args.save_agent)
            )
            with trainer:
                agent = trainer.train()

                if config.input.backtest.run:
                    logger.info('Starting back testing...')

                    logger.info('Trainval split:')
                    train_env = build_env(config, dataset, mode=Mode.BacktestTrain)
                    evaluation.backtest(
                        train_env,
                        agent,
                        deterministic=config.input.backtest.deterministic,
                        name='trainval_backtest'
                    )

                    logger.info('Test split:')
                    dataset = build_dataset(config, storage_dir, mode=Mode.Backtest)
                    test_env = build_env(config, dataset, mode=Mode.Backtest)
                    evaluation.backtest(
                        test_env,
                        agent,
                        deterministic=config.input.backtest.deterministic,
                        name='test_backtest'
                    )
                    test_env.close()

                dataset.close()
                train_env.close()
                val_env.close()
        elif mode == Mode.Backtest:
            logger.info('Starting back testing...')

            logger.info('Trainval split:')
            trainval_dataset = build_dataset(config, storage_dir, mode=mode.Backtest)
            trainval_env = build_env(config, trainval_dataset, mode=Mode.BacktestTrain)
            agent = build_agent(
                config,
                trainval_env,
                storage_dir,
                resume=True,
                agent_path=args.agent_path
            )
            evaluation.backtest(
                trainval_env,
                agent,
                deterministic=config.backtest.deterministic,
                name='trainval_backtest'
            )

            logger.info('Test split:')
            test_dataset = build_dataset(config, storage_dir, mode=Mode.Backtest)
            test_env = build_env(config, test_dataset, mode=Mode.Backtest)
            agent = build_agent(
                config,
                test_env,
                storage_dir,
                resume=True,
                agent_path=args.agent_path
            )
            evaluation.backtest(
                test_env,
                agent,
                deterministic=config.backtest.deterministic,
                name='test_backtest'
            )

            trainval_dataset.close()
            trainval_env.close()
            test_dataset.close()
            test_env.close()
        elif mode == Mode.Baseline:
            raise NotImplementedError()
