import logging
import pprint

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
            logger.info(f'Backtest info:\n')
            logger.info(pprint.pformat(info, indent=4))

            break

    if render_all:
        episode_metrics = info['episode_metrics']
        title = f'SR={round(episode_metrics["sharpe_ratio"], 4)};' \
                f'Total Assets={round(info["total_assets"], 4)};' \
                f'Annual Return={round(episode_metrics["annual_return"], 4)}'
        env.render_all(title=title, name=f'{name}.png')

    return env.create_report()
