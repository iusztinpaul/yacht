from .enums import *
from .base import *
from .day import *

from .action_schemas import build_action_schema
from .reward_schemas import build_reward_schema

import gym
from gym.envs.registration import register

from .wrappers import MultiFrequencyDictToBoxWrapper
from ..config.proto.environment_pb2 import EnvironmentConfig

logger = logging.getLogger(__file__)

environment_registry = {
    'DayForecastEnv': DayForecastEnv
}


def build_env(env_config: EnvironmentConfig, dataset: TradingDataset):
    reward_schema = build_reward_schema(env_config)
    action_schema = build_action_schema(env_config)

    env = gym.make(
        env_config.name,
        dataset=dataset,
        reward_schema=reward_schema,
        action_schema=action_schema
    )
    env = MultiFrequencyDictToBoxWrapper(env)

    return env


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
