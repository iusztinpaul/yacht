from .enums import *
from .base import *
from .day import *

from .action_schemas import build_action_schema
from .monitors import RewardRendererMonitor
from .reward_schemas import build_reward_schema

import gym
from gym.envs.registration import register

from .wrappers import MultipleTimeFrameDictToBoxWrapper, WandBWrapper
from .. import utils
from ..config import Config
from ..config.proto.environment_pb2 import EnvironmentConfig

logger = logging.getLogger(__file__)

environment_registry = {
    'DayForecastEnv': DayForecastEnv
}


def build_env(config: Config, dataset: TradingDataset, mode: Mode):
    env_config: EnvironmentConfig = config.environment

    action_schema = build_action_schema(config)
    reward_schema = build_reward_schema(
        config, max_score=utils.compute_max_score(
            num_days=dataset.num_days,
            action_max_score=action_schema.max_units_per_asset
        )
    )

    env_kwargs = {
        'dataset': dataset,
        'reward_schema': reward_schema,
        'action_schema': action_schema
    }
    if env_config.name == 'SingleAssetTradingEnvironment-v0':
        env_kwargs.update({
            'buy_commission': env_config.buy_commission,
            'sell_commission': env_config.sell_commission,
            'initial_cash_position': env_config.initial_cash_position
        })

    env = gym.make(
        env_config.name,
        **env_kwargs
    )
    env = MultipleTimeFrameDictToBoxWrapper(env)
    env = RewardRendererMonitor(
        final_step=config.train.collecting_n_steps * config.train.collect_n_times,
        storage_dir=dataset.storage_dir,
        env=env,
    )
    if utils.get_experiment_tracker_name(dataset.storage_dir) == 'wandb':
        env = WandBWrapper(
            env=env,
            mode=mode
        )

    return env


def register_gym_envs():
    to_register_envs = {
        'DayForecastEnv-v0': {
            'entry_point': 'yacht.environments.day:DayForecastEnv',
            'kwargs': {
            }
        },
        'SingleAssetTradingEnvironment-v0': {
            'entry_point': 'yacht.environments.single_asset:SingleAssetTradingEnvironment',
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
