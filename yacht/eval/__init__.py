import logging

import matplotlib.pyplot as plt


logger = logging.getLogger(__file__)


def run_agent(env, agent):
    observation = env.reset()
    while True:
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        if done:
            logger.info("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()
