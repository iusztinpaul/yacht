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
        logger.info(f'Starting training for {config.train.episodes} episodes.')
        for i in range(config.train.episodes // config.train.k_fold_splits):
            k_fold = build_k_fold(config.input, config.train)
            for k, (train_indices, val_indices) in enumerate(k_fold.split(X=dataset.get_folding_values())):
                episode = i * config.train.k_fold_splits + k
                logger.info(f'Episode - {episode}')

                # TODO: Is it ok to slide the window between 2 splits of train intervals ? Data not contiguous
                train_dataset = build_dataset_wrapper(dataset, indices=train_indices)
                val_dataset = build_dataset_wrapper(dataset, indices=val_indices)

                train_env.set_dataset(train_dataset)
                val_env.set_dataset(val_dataset)

                agent = agent.learn(
                    total_timesteps=1,
                    callback=None,
                    tb_log_name=config.input.env,
                    log_interval=1,
                    eval_env=val_env,
                    eval_freq=len(train_indices),
                    n_eval_episodes=1,
                    eval_log_path=None,
                    reset_num_timesteps=True
                )

    if config.meta.back_test:
        logger.info('Starting back testing...')
        dataset = build_dataset(config.input, config.train, storage_path, mode='test')
        back_test_env = build_env(config.input, dataset)

        back_testing.run_agent(back_test_env, agent)

        back_test_env.close()

    train_env.close()
    val_env.close()
