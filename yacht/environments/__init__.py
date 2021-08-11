from gym.wrappers import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv

from .enums import *
from .base import *
from .day import *

from .action_schemas import build_action_schema
from .reward_schemas import build_reward_schema

import gym
from gym.envs.registration import register

from yacht.logger import Logger
from .wrappers import MultiFrequencyDictToBoxWrapper, MetricsVecEnvWrapper
from .. import utils, Mode
from ..config import Config
from ..config.proto.environment_pb2 import EnvironmentConfig


def build_env(
        config: Config,
        dataset: ChooseAssetDataset,
        logger: Logger,
        mode: Mode,
) -> MetricsVecEnvWrapper:
    def _wrappers(env: Union[Monitor, BaseAssetEnvironment]) -> gym.Env:
        if isinstance(env, Monitor):
            assert isinstance(env.env, BaseAssetEnvironment), f'Wrong env type: {type(env.env)}.'

        # Classic methods can handle directly a dict for simplicity.
        if not config.agent.is_classic_method:
            env = MultiFrequencyDictToBoxWrapper(env)

        return env

    env_config: EnvironmentConfig = config.environment
    backtest_config = config.input.backtest

    action_schema = build_action_schema(config)
    reward_schema = build_reward_schema(
        config, max_score=utils.compute_max_score(
            num_days=dataset.num_days,
            action_max_score=action_schema.action_scaling_factor
        )
    )
    env_kwargs = {
        'name': mode.value,
        'dataset': dataset,
        'reward_schema': reward_schema,
        'action_schema': action_schema,
        'render_on_done': not mode.is_trainable()
    }
    if env_config.name == 'SingleAssetEnvironment-v0':
        env_kwargs.update({
            'buy_commission': env_config.buy_commission,
            'sell_commission': env_config.sell_commission,
            'initial_cash_position': env_config.initial_cash_position
        })

    n_envs = env_config.n_envs if mode.is_trainable() else len(backtest_config.tickers) * backtest_config.n_runs
    env = make_vec_env(
        env_id=env_config.name,
        n_envs=n_envs,
        seed=0,
        start_index=0,
        monitor_dir=utils.build_monitor_dir(dataset.storage_dir, mode),
        wrapper_class=_wrappers,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if env_config.envs_on_different_processes else DummyVecEnv,
        vec_env_kwargs=None,
        monitor_kwargs=None,
        wrapper_kwargs=None
    )

    if utils.get_experiment_tracker_name(dataset.storage_dir) == 'wandb':
        env = MetricsVecEnvWrapper(
            env=env,
            logger=logger,
            mode=mode
        )

    return env


def register_gym_envs():
    to_register_envs = {
        'DayForecastEnvironment-v0': {
            'entry_point': 'yacht.environments.day:DayForecastEnvironment',
            'kwargs': {
            }
        },
        'SingleAssetEnvironment-v0': {
            'entry_point': 'yacht.environments.single_asset:SingleAssetEnvironment',
            'kwargs': {
            }
        }
    }

    gym_env_dict = gym.envs.registration.registry.env_specs

    for env_id, parameters in to_register_envs.items():
        if env_id not in gym_env_dict:
            register(
                id=env_id,
                entry_point=parameters['entry_point'],
                kwargs=parameters['kwargs']
            )
