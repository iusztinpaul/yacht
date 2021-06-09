import logging

import matplotlib.pyplot as plt


logger = logging.getLogger(__file__)


def run_agent(env, agent, render: bool = True, render_all=False):
    assert render is not True or render_all is not True, \
        'Either render on the fly or in the end.'

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
        plt.cla()
        env.render_all()
        plt.show()

    env.save_rendering(name='back_test.png')
