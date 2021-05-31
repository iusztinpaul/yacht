import logging

import matplotlib.pyplot as plt


logger = logging.getLogger(__file__)


def run_agent(env, agent, render: bool = True):
    observation = env.reset()
    while True:
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            logger.info(f'Back test info:\n{info}')
            break

    if render:
        plt.cla()
        env.render_all()
        plt.show()
