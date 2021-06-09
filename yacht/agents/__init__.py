from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from torch import nn

from yacht import utils
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


def build_agent(config, env: TradingEnv, resume: bool = False, agent_path: str = None) -> BaseAlgorithm:
    agent_config = config.agent
    backbone_config = agent_config.backbone
    feature_extractor_config = agent_config.feature_extractor
    input_config = config.input
    train_config = config.train

    # The agent is the main wrapper over all the logic.
    agent_class = agents_registry[agent_config.name]
    if resume:
        assert agent_path is not None

        return agent_class.load(agent_path)
    else:
        # The agent has a policy.
        policy_class = GenericActorCriticPolicy
        # The policy will build the feature_extractor & the backbone.proto.
        feature_extractor_class = feature_extractor_registry[agent_config.feature_extractor.name]
        backbone_class = backbone_registry[agent_config.backbone.name]
        policy_kwargs = {
            'activation_fn': nn.ReLU,
            'backbone_class': backbone_class,
            'backbone_kwargs': {
                'features_dim': backbone_config.input_features_dim,
                'window_size': input_config.window_size,
                'intervals': input_config.intervals
            },
            'features_extractor_class': feature_extractor_class,
            'features_extractor_kwargs': {
                'features_dim': feature_extractor_config.output_features_dim,
                'intervals': input_config.intervals,
                'features': input_config.features
            }
        }

        train_val_num_days = utils.get_train_val_num_days(
            input_config.start,
            input_config.end,
            input_config.back_test_split_ratio,
            train_config.k_fold_embargo_ratio
        )
        train_val_num_days = train_val_num_days - train_val_num_days % train_config.batch_size
        assert train_val_num_days > 0

        # TODO: Look over all Agents hyper-parameters
        return agent_class(
            policy=policy_class,
            env=env,
            verbose=agent_config.verbose,
            learning_rate=train_config.learning_rate,
            n_steps=train_val_num_days,
            batch_size=train_config.batch_size,
            n_epochs=train_config.n_epochs,
            policy_kwargs=policy_kwargs
        )




