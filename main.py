import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

import yacht.agents.predict
from yacht.config import load_config, export_config
from yacht.data.datasets import build_dataset
from yacht.environments import build_env
from yacht import utils, evaluation
from yacht import environments
from yacht.agents import build_agent
from yacht.trainer import build_trainer

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
    log_dir = utils.setup_logger(
        level=args.logger_level,
        storage_dir=storage_dir
    )
    environments.register_gym_envs()
    config = load_config(os.path.join(ROOT_DIR, 'yacht', 'config', 'configs', args.config_file_name))
    export_config(config, storage_dir)
    logger.info(f'Config:\n{config}')

    if args.mode == 'train':
        dataset = build_dataset(config, storage_dir, mode='trainval')
        train_env = build_env(config, dataset)
        val_env = build_env(config, dataset)
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

            if config.meta.back_test:
                logger.info('Starting back testing...')
                evaluation.backtest(
                    train_env,
                    agent,
                    render=False,
                    render_all=True,
                    name='after_train_trainval_backtest'
                )
                dataset = build_dataset(config, storage_dir, mode='test')
                test_env = build_env(config, dataset)
                evaluation.backtest(
                    test_env,
                    agent,
                    render=False,
                    render_all=True,
                    name='after_train_test_backtest'
                )
                test_env.close()

            dataset.close()
            train_env.close()
            val_env.close()
    elif args.mode == 'backtest':
        logger.info('Starting back testing...')

        trainval_dataset = build_dataset(config, storage_dir, mode='trainval')
        trainval_env = build_env(config, trainval_dataset)
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
            render=False,
            render_all=True,
            name='after_backtest_trainval_split_backtest'
        )

        test_dataset = build_dataset(config, storage_dir, mode='test')
        test_env = build_env(config, test_dataset)
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
            render=False,
            render_all=True,
            name='after_backtest_test_split_backtest'
        )

        trainval_dataset.close()
        trainval_env.close()
        test_dataset.close()
        test_env.close()
    elif args.mode == 'max_possible_profit':
        dataset = build_dataset(config, storage_dir, mode='test')
        test_env = build_env(config, dataset)
        test_env.max_possible_profit(stateless=False)
        test_env.render_all(name='max_possible_profit.png')
        plt.show()

        dataset.close()
        test_env.close()
