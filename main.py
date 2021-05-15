import argparse
import logging
import os

import matplotlib.pyplot as plt

from yacht.config import load_config
from yacht.environments import build_env
from yacht import utils
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

    env = build_env(config.input, storage_path)
    agent = build_agent(config, env)

    if args.mode == 'train':
        agent.learn(config.train.timestamps)

    observation = env.reset()
    while True:
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()
