from .enums import *
from .base import *
from .day import *

import gym
from gym.envs.registration import register

from ..config import InputConfig

logger = logging.getLogger(__file__)

environment_registry = {
    'DayForecastEnv': DayForecastEnv
}


def build_env(input_config: InputConfig, dataset: TradingDataset):
    return gym.make(
        input_config.env,
        dataset=dataset,
    )


def register_gym_envs():
    to_register_envs = {
        'DayForecastEnv-v0': {
            'entry_point': 'yacht.environments.day:DayForecastEnv',
            'kwargs': {
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
