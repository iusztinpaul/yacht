from gym.wrappers import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .base import *
from .day import *

from yacht.environments.action_schemas import build_action_schema
from yacht.environments.reward_schemas import build_reward_schema

import gym
from gym.envs.registration import register

from yacht.logger import Logger
from yacht.environments.wrappers import MultiFrequencyDictToBoxWrapper, MetricsVecEnvWrapper
from yacht import utils, Mode
from yacht.config import Config, EnvironmentConfig


def build_env(
        config: Config,
        dataset: SampleAssetDataset,
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

    action_schema = build_action_schema(config, dataset, mode)
    reward_schema = build_reward_schema(config)
    env_kwargs = {
        'name': mode.value,
        'dataset': dataset,
        'reward_schema': reward_schema,
        'action_schema': action_schema,
        'compute_metrics': not mode.is_trainable(),
        'buy_commission': env_config.buy_commission,
        'sell_commission': env_config.sell_commission,
        'initial_cash_position': env_config.initial_cash_position,
        'include_weekends': config.input.include_weekends
    }

    if mode.is_trainable() or config.input.backtest.deterministic is False:
        n_envs = env_config.n_envs
    else:
        # If we build a environment for backtest & it is deterministic it is redundant to use multiple envs.
        n_envs = 1
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
    env = MetricsVecEnvWrapper(
        venv=env,
        n_metrics_episodes=len(dataset.datasets),
        logger=logger,
        mode=mode,
        metrics_to_log=list(config.meta.metrics_to_log),
        extra_stats_metrics=list(config.meta.metrics_to_save_best_on)
    )

    return env


def register_gym_envs():
    to_register_envs = {
        'MultiAssetEnvironment-v0': {
            'entry_point': 'yacht.environments.multi_asset:MultiAssetEnvironment',
            'kwargs': {
            }
        },
        'OrderExecutionEnvironment-v0': {
            'entry_point': 'yacht.environments.order_execution:OrderExecutionEnvironment',
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
