import argparse
import logging
import os


from yacht.config import load_config
from yacht.environments import build_train_val_env, build_back_test_env
from yacht import utils, back_testing
from yacht import environments
from yacht.agents import build_agent

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
    utils.setup_logger(
        level=args.logger_level,
        storage_path=storage_path
    )
    environments.register_gym_envs()
    config = load_config(os.path.join(ROOT_DIR, 'yacht', 'config', 'configs', args.config_file))
    logger.info(f'Config:\n{config}')

    train_env = build_train_val_env(config.input, storage_path, mode='train')
    val_env = build_train_val_env(config.input, storage_path, mode='val')
    agent = build_agent(config, train_env)

    if args.mode == 'train':
        logger.info('Started training...')
        agent = agent.learn(
            config.train.steps,
            log_interval=config.train.log_freq,
            eval_env=val_env,
            eval_freq=config.train.val_freq
        )

    if config.meta.back_test:
        logger.info('Started back testing...')
        back_test_env = build_back_test_env(config.input, storage_path)
        back_testing.run_agent(back_test_env, agent)
        back_test_env.close()

    train_env.close()
    val_env.close()
