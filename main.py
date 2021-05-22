import argparse
import logging
import os


from yacht.config import load_config
from yacht.data.datasets import build_dataset, build_dataset_wrapper, IndexedDatasetMixin
from yacht.data.k_fold import build_k_fold
from yacht.environments import build_env
from yacht import utils, back_testing
from yacht import environments
from yacht.agents import build_agent
from yacht.trainer import build_trainer

logger = logging.getLogger(__file__)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=('train', ))
parser.add_argument("--config_file", required=True, help='Path to your *.yaml configuration file.')
parser.add_argument("--resume_training", default=False, action='store_true', help='Resume training or not.')
parser.add_argument("--storage_path", required=True, help='Directory where your model & logs will be saved.')
parser.add_argument("--logger_level", default='info', choices=('info', 'debug', 'warn'))


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
    config = load_config(os.path.join(ROOT_DIR, 'yacht', 'config', 'configs', args.config_file))
    logger.info(f'Config:\n{config}')

    dataset = build_dataset(config.input, config.train, storage_path, mode='trainval')
    train_env = build_env(config.input, dataset)
    val_env = build_env(config.input, dataset)
    agent = build_agent(config, train_env)

    if args.mode == 'train':
        trainer = build_trainer(
            config=config,
            agent=agent,
            dataset=dataset,
            train_env=train_env,
            val_env=val_env
        )
        trainer.train(config=config)

        if config.meta.back_test:
            logger.info('Starting back testing...')
            dataset = build_dataset(config.input, config.train, storage_path, mode='test')
            back_test_env = build_env(config.input, dataset)

            back_testing.run_agent(back_test_env, agent)

            back_test_env.close()

    train_env.close()
    val_env.close()
