import logging
import os
from typing import Union

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch import nn

from yacht import utils
from yacht.agents.modules.multi_frequency import MultiFrequencyFeatureExtractor
from yacht.agents.policies.generic import GenericActorCriticPolicy
from yacht.config.proto.net_architecture_pb2 import NetArchitectureConfig
from yacht.environments import TradingEnv


logger = logging.getLogger(__file__)


agents_registry = {
    'PPO': PPO
}

policy_registry = {
    'MlpPolicy': 'MlpPolicy'
}

feature_extractor_registry = {
    'MultiFrequencyFeatureExtractor': MultiFrequencyFeatureExtractor
}

activation_fn_registry = {
    'ReLU': nn.ReLU
}


def build_agent(
        config,
        env: TradingEnv,
        storage_dir: str,
        resume: bool = False,
        agent_path: str = None
) -> BaseAlgorithm:
    agent_config = config.agent
    policy_config = config.agent.policy
    feature_extractor_config = policy_config.feature_extractor
    input_config = config.input
    train_config = config.train

    # The agent is the main wrapper over all the logic.
    agent_class = agents_registry[agent_config.name]
    if resume:
        if agent_path is None:
            agent_path = utils.build_last_checkpoint_path(env.dataset.storage_dir)
            logger.info(f'Resuming from the last checkpoint: {agent_path}')

            assert os.path.exists(agent_path)

        return agent_class.load(agent_path)
    else:
        # The agent has a policy.
        policy_class = policy_registry[policy_config.name]
        # The policy will build the feature_extractor.
        feature_extractor_class = feature_extractor_registry[policy_config.feature_extractor.name]
        activation_fn_class = activation_fn_registry[policy_config.activation_fn]
        policy_kwargs = {
            'net_arch': _build_net_arch_dict(policy_config.net_arch, agent_class),
            'activation_fn': activation_fn_class,
            'features_extractor_class': feature_extractor_class,
            'features_extractor_kwargs': {
                'features_dim': list(feature_extractor_config.features_dim),
                'activation_fn': activation_fn_class,
                'window_size': input_config.window_size,
                'intervals': list(input_config.intervals),
                'features': list(input_config.features) + list(input_config.technical_indicators),
                'env_features_len': env.observation_env_features_len,
                'drop_out_p': feature_extractor_config.drop_out_p
            }
        }

        return agent_class(
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(storage_dir, 'tensorboard'),
            device='cuda' if config.meta.device == 'gpu' else config.meta.device
        )


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


