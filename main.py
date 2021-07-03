import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from yacht.config import load_config
from yacht.data.datasets import build_dataset
from yacht.environments import build_env
from yacht import utils, back_testing
from yacht import environments
from yacht.agents import build_agent
from yacht.trainer import build_trainer

logger = logging.getLogger(__file__)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('train', 'back_test', 'max_possible_profit'))
parser.add_argument(
    '--config_file_name',
    required=True,
    help='Name of the *.config file from the configuration dir.'
)
parser.add_argument('--resume_training', default=False, action='store_true', help='Resume training or not.')
parser.add_argument('--storage_path', required=True, help='Directory where your model & logs will be saved.')
parser.add_argument('--logger_level', default='info', choices=('info', 'debug', 'warn'))


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = parser.parse_args()
    if '.' == args.storage_path[0]:
        storage_path = os.path.join(ROOT_DIR, args.storage_path[2:])
    else:
        storage_path = args.storage_path

    utils.load_env(root_dir=ROOT_DIR)
    log_dir = utils.setup_logger(
        level=args.logger_level,
        storage_path=storage_path
    )
    environments.register_gym_envs()
    config = load_config(os.path.join(ROOT_DIR, 'yacht', 'config', 'configs', args.config_file_name))
    logger.info(f'Config:\n{config}')

    if args.mode == 'train':
        dataset = build_dataset(config, storage_path, mode='trainval')
        train_env = build_env(config, dataset)
        val_env = build_env(config, dataset)
        agent = build_agent(config, train_env, storage_path)

        trainer = build_trainer(
            config=config,
            agent=agent,
            dataset=dataset,
            train_env=train_env,
            val_env=val_env
        )
        with trainer:
            agent = trainer.train()
            agent.save(os.path.join(storage_path, 'agent'))

            if config.meta.back_test:
                logger.info('Starting back testing...')
                back_testing.run_agent(
                    train_env,
                    agent,
                    render=False,
                    render_all=True,
                    name='trainval_split_backtest'
                )
                dataset = build_dataset(config, storage_path, mode='test')
                test_env = build_env(config, dataset)
                back_testing.run_agent(
                    test_env,
                    agent,
                    render=False,
                    render_all=True,
                    name='test_split_backtest'
                )
                test_env.close()

            dataset.close()
            train_env.close()
            val_env.close()
    elif args.mode == 'back_test':
        logger.info('Starting back testing...')

        trainval_dataset = build_dataset(config, storage_path, mode='trainval')
        trainval_env = build_env(config, trainval_dataset)
        agent = build_agent(
            config,
            trainval_env,
            storage_path,
            resume=True,
            agent_file=os.path.join(storage_path, 'agent')
        )
        back_testing.run_agent(
            trainval_env,
            agent,
            render=True,
            render_all=False,
            name='trainval_split_backtest'
        )

        test_dataset = build_dataset(config, storage_path, mode='test')
        test_env = build_env(config, test_dataset)
        agent = build_agent(
            config,
            test_env,
            storage_path,
            resume=True,
            agent_file=os.path.join(storage_path, 'agent')
        )
        back_testing.run_agent(
            test_env,
            agent,
            render=True,
            render_all=False,
            name='test_split_backtest'
        )

        trainval_dataset.close()
        trainval_env.close()
        test_dataset.close()
        test_env.close()
    elif args.mode == 'max_possible_profit':
        dataset = build_dataset(config, storage_path, mode='test')
        test_env = build_env(config, dataset)
        test_env.max_possible_profit(stateless=False)
        test_env.render_all(name='max_possible_profit.png')
        plt.show()

        dataset.close()
        test_env.close()
