import logging
from .enums import *
from .base import *
from .day import *

import gym
from gym.envs.registration import register

from yacht.data.datasets import build_train_val_dataset, build_back_test_dataset
from .enums import *
from ..config import InputConfig
from ..data.normalizers import build_normalizer

logger = logging.getLogger(__file__)

environment_registry = {
    'DayForecastEnv': DayForecastEnv
}


def build_train_val_env(input_config: InputConfig, storage_path: str, mode: str):
    dataset = build_train_val_dataset(input_config, storage_path, mode)
    normalizer = build_normalizer(input_config.env_normalizer)

    return gym.make(
        input_config.env,
        dataset=dataset,
        normalizer=normalizer,
        window_size=input_config.window_size
    )


def build_back_test_env(input_config: InputConfig, storage_path: str):
    dataset = build_back_test_dataset(input_config, storage_path)
    normalizer = build_normalizer(input_config.env_normalizer)

    return gym.make(
        input_config.env,
        dataset=dataset,
        normalizer=normalizer,
        window_size=input_config.window_size
    )


def register_gym_envs():
    to_register_envs = {
        'DayForecastEnv-v0': {
            'entry_point': 'yacht.environments.day:DayForecastEnv',
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
