import os
from typing import Union

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch import nn

from yacht.agents.modules.day import MultipleTimeFramesFeatureExtractor, DayForecastNetwork
from yacht.agents.policies.generic import GenericActorCriticPolicy
from yacht.config.proto.net_architecture_pb2 import NetArchitectureConfig
from yacht.environments import TradingEnv

agents_registry = {
    'PPO': PPO
}

policy_registry = {
    'MlpPolicy': 'MlpPolicy'
}

feature_extractor_registry = {
    'MultipleTimeFramesFeatureExtractor': MultipleTimeFramesFeatureExtractor
}

activation_fn_registry = {
    'ReLU': nn.ReLU
}


def build_agent(
        config, env: TradingEnv,
        storage_path: str,
        resume: bool = False,
        agent_file: str = None
) -> BaseAlgorithm:
    agent_config = config.agent
    policy_config = config.agent.policy
    feature_extractor_config = policy_config.feature_extractor
    input_config = config.input
    train_config = config.train

    # The agent is the main wrapper over all the logic.
    agent_class = agents_registry[agent_config.name]
    if resume:
        assert agent_file is not None
        agent_path = os.path.join(storage_path, agent_file)

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
                'features_dim': feature_extractor_config.output_features_dim,
                'activation_fn': activation_fn_class,
                'intervals': input_config.intervals,
                'features': list(input_config.features) + list(input_config.technical_indicators)
            }
        }

        # TODO: Look over all Agents hyper-parameters
        return agent_class(
            policy=policy_class,
            env=env,
            verbose=agent_config.verbose,
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
            tensorboard_log=storage_path,
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


