import logging

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from yacht.environments import TradingEnv


logger = logging.getLogger(__file__)


def run_agent(
        env: TradingEnv,
        agent: BaseAlgorithm,
        render: bool = True,
        render_all: bool = False,
        name: str = 'backtest'
) -> pd.DataFrame:
    assert render is not True or render_all is not True, \
        'Either render on the fly or in the end.'

    agent.policy.eval()

    observation = env.reset()
    while True:
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        if render:
            env.render()
        if done:
            logger.info(f'Back test info:\n{info}')
            break

    if render_all:
        title = f'{info["total_value_completeness"]*100}%'
        env.render_all(title=title, name=f'{name}.png')

    return env.create_report()
