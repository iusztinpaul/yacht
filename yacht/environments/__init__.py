import logging
from .enums import *
from .base import *
from .day import *

import gym
from gym.envs.registration import register

from yacht.data.datasets import build_dataset
from .enums import *


logger = logging.getLogger(__file__)

environment_registry = {
    'DayForecastEnv': DayForecastEnv
}


def build_env(input_config, storage_path):
    dataset = build_dataset(input_config, storage_path)
    env_class = environment_registry[input_config.env]

    return env_class(
        dataset=dataset,
        window_size=input_config.window_size
    )


def register_gym_envs():
    to_register_envs = {
        'day.config.txt-forecast-v0': {
            'entry_point': 'yacht.environments.day.config.txt:DayForecastEnv',
            'kwargs': {
                'window_size': 14
            }
        }
    }

    gym_env_dict = gym.envs.registration.registry.env_specs

    for env_id, parameters in to_register_envs.items():
        # TODO: Find a better way to fix multiple 'register' attempts
        if env_id in gym_env_dict:
            logger.info('Remove {} from registry'.format(env_id))
            del gym.envs.registration.registry.env_specs[env_id]

        register(
            id=env_id,
            entry_point=parameters['entry_point'],
            kwargs=parameters['kwargs']
        )
