import argparse
import logging
import os

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
    args = parser.parse_args()
    if '.' in args.storage_path:
        storage_path = os.path.join(ROOT_DIR, args.storage_path)
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
        train_env = build_env(config.environment, dataset)
        val_env = build_env(config.environment, dataset)
        agent = build_agent(config, train_env)

        trainer = build_trainer(
            config=config,
            agent=agent,
            dataset=dataset,
            train_env=train_env,
            val_env=val_env
        )
        agent = trainer.train()
        # agent.save(os.path.join(storage_path, 'agent'))

        if config.meta.back_test:
            logger.info('Starting back testing...')
            dataset = build_dataset(config, storage_path, mode='test')

            back_test_env = build_env(config.environment, dataset)
            back_testing.run_agent(back_test_env, agent, render=False, render_all=True)
            back_test_env.close()

        dataset.close()
        train_env.close()
        val_env.close()
        trainer.close()
    elif args.mode == 'back_test':
        logger.info('Starting back testing...')

        dataset = build_dataset(config, storage_path, mode='test')
        back_test_env = build_env(config.environment, dataset)
        agent = build_agent(
            config,
            back_test_env,
            resume=True,
            agent_path=os.path.join(storage_path, 'agent')
        )

        back_testing.run_agent(back_test_env, agent, render=True, render_all=False)

        dataset.close()
        back_test_env.close()
    elif args.mode == 'max_possible_profit':
        dataset = build_dataset(config, storage_path, mode='test')
        back_test_env = build_env(config.environment, dataset)
        back_test_env.max_possible_profit()
        back_test_env.render_all(name='max_possible_profit.png')
        plt.show()

        dataset.close()
        back_test_env.close()
