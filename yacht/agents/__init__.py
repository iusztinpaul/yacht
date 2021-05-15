from stable_baselines3 import PPO
from torch import nn

from yacht.agents.modules.day import MultipleTimeFramesFeatureExtractor, DayForecastNetwork
from yacht.agents.policies.generic import GenericActorCriticPolicy
from yacht.environments import TradingEnv

agents_registry = {
    'PPO': PPO
}


feature_extractor_registry = {
    'MultipleTimeFramesFeatureExtractor': MultipleTimeFramesFeatureExtractor
}


backbone_registry = {
    'DayForecastNetwork': DayForecastNetwork
}


def build_agent(config, env: TradingEnv):
    agent_config = config.agent
    backbone_config = agent_config.backbone
    feature_extractor_config = agent_config.feature_extractor
    input_config = config.input
    train_config = config.train

    # The agent is the main wrapper over all the logic.
    agent_class = agents_registry[agent_config.name]
    # The agent has a policy.
    policy_class = GenericActorCriticPolicy
    # The policy will build the feature_extractor & the backbone.proto.
    feature_extractor_class = feature_extractor_registry[agent_config.feature_extractor.name]
    backbone_class = backbone_registry[agent_config.backbone.name]
    policy_kwargs = {
        'activation_fn': nn.ReLU,
        'backbone_class': backbone_class,
        'backbone_kwargs': {
            'features_dim': backbone_config.features_dim,
            'window_size': input_config.window_size,
        },
        'features_extractor_class': feature_extractor_class,
        'features_extractor_kwargs': {
            'features_dim': feature_extractor_config.features_dim,
            'intervals': input_config.intervals
        }
    }

    # TODO: Look over all Agents hyper-parameters
    return agent_class(
        policy=policy_class,
        env=env,
        verbose=agent_config.verbose,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        n_epochs=train_config.n_epochs,
        policy_kwargs=policy_kwargs
    )




