from .agents import *

import os
from typing import Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from yacht import utils, Mode
from yacht.logger import Logger
from yacht.agents.classic import BuyAndHoldAgent, BaseClassicAgent, DCFAgent
from yacht.agents.modules import RecurrentFeatureExtractor, MultiFrequencyFeatureExtractor
from yacht.config import Config
from yacht.config.proto.net_architecture_pb2 import NetArchitectureConfig
from ..environments import BaseAssetEnvironment

reinforcement_learning_agents = {
    'PPO': PPO,
    'SAC': SAC
}
classic_agents = {
    'BuyAndHold': BuyAndHoldAgent,
    'DCF': DCFAgent
}
agents_registry = {**reinforcement_learning_agents, **classic_agents}

policy_registry = {
    'MlpPolicy': 'MlpPolicy'
}

feature_extractor_registry = {
    'MultiFrequencyFeatureExtractor': MultiFrequencyFeatureExtractor,
    'RecurrentFeatureExtractor': RecurrentFeatureExtractor,
    '': None,
    None: None
}

activation_fn_registry = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh
}


def build_agent(
        config: Config,
        env: Union[BaseAssetEnvironment, VecEnv],
        logger: Logger,
        storage_dir: str,
        resume: bool = False,
        agent_from: str = None,
        best_metric: str = None
) -> BaseAlgorithm:
    """

    Args:
        config:
        env:
        logger:
        storage_dir:
        resume:
        agent_from: choose from (latest checkpoint, best checkpoint, absolute_path to the checkpoint)
        best_metric: The metric you want to resume the agent from. If it is none it will be resumed based on the best
            reward. You have to choose `agent_from=best`.

    Returns:

    """

    agent_config = config.agent
    policy_config = config.agent.policy
    feature_extractor_config = policy_config.feature_extractor
    input_config = config.input
    train_config = config.train

    # The agent is the main wrapper over all the logic.
    agent_class = agents_registry[agent_config.name]
    if agent_config.is_classic_method:
        return agent_class(
            env=env
        )

    if resume:
        if agent_from == 'best':
            if best_metric is None:
                agent_from = utils.build_best_reward_checkpoint_path(storage_dir, Mode.FineTuneTrain)
                logger.info(f'Resuming from the best reward checkpoint: {agent_from}')
            else:
                agent_from = utils.build_best_metric_checkpoint_path(storage_dir, Mode.FineTuneTrain, best_metric)
                logger.info(f'Resuming from the best metric - {best_metric} - checkpoint: {agent_from}')
        elif agent_from == 'latest':
            agent_from = utils.build_last_checkpoint_path(storage_dir, Mode.FineTuneTrain)
            logger.info(f'Resuming from the latest checkpoint: {agent_from}')

        assert os.path.exists(agent_from)

        return agent_class.load(agent_from)
    else:
        # The agent has a policy.
        policy_class = policy_registry[policy_config.name]
        # The policy will build the feature_extractor.
        feature_extractor_class = feature_extractor_registry[policy_config.feature_extractor.name]
        activation_fn_class = activation_fn_registry[policy_config.activation_fn]
        policy_kwargs = {
            'net_arch': _build_net_arch_dict(policy_config.net_arch, agent_class),
            'activation_fn': activation_fn_class,
        }
        if feature_extractor_class:
            policy_kwargs['features_extractor_class'] = feature_extractor_class
            policy_kwargs['features_extractor_kwargs'] = {
                'features_dim': list(feature_extractor_config.features_dim),
                'activation_fn': activation_fn_class,
                'window_size': input_config.window_size,
                'intervals': list(input_config.intervals),
                'features': list(input_config.features) + list(input_config.technical_indicators),
                'env_features_len': env.envs[0].observation_env_features_len,
                'num_assets': input_config.num_assets_per_dataset,
                'drop_out_p': feature_extractor_config.drop_out_p,
                # TODO: Remove rnn_layer_type config parameter
                'rnn_layer_type': nn.GRU if feature_extractor_config.rnn_layer_type == 'GRU' else nn.LSTM
            }

        agent = agent_class(
            policy=policy_class,
            env=env,
            verbose=1 if agent_config.verbose else 0,
            learning_rate=train_config.learning_rate,
            batch_size=train_config.batch_size,
            n_steps=train_config.collecting_n_steps,
            n_epochs=train_config.n_epochs,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            clip_range=train_config.clip_range,
            ent_coef=train_config.entropy_coefficient,
            vf_coef=train_config.vf_coefficient,
            max_grad_norm=train_config.max_grad_norm,
            use_sde=train_config.use_sde,
            sde_sample_freq=train_config.sde_sample_freq,
            policy_kwargs=policy_kwargs,
            device='cuda' if config.meta.device == 'gpu' else config.meta.device
        )
        agent.set_logger(logger)

        return agent


def _build_net_arch_dict(net_arch: NetArchitectureConfig, agent_class: type) -> Union[list, dict]:
    is_policy_based = OnPolicyAlgorithm in agent_class.mro()
    structured_net_arch = [] if is_policy_based else dict()

    if is_policy_based:
        structured_net_arch.extend(net_arch.shared)
        structured_net_arch.append({
            'vf': list(net_arch.vf),
            'pi': list(net_arch.pi)
        })
    else:
        structured_net_arch['qf'] = list(net_arch.qf)
        structured_net_arch['pi'] = list(net_arch.pi)

    return structured_net_arch


