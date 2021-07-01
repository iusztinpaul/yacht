import matplotlib.pyplot as plt
import numpy as np

MAX_VALUE = 100


def f(x, max_score_value=100):
    return (2 * max_score_value) / (x + max_score_value)


if __name__ == '__main__':
    scores = np.arange(start=-MAX_VALUE+1, stop=MAX_VALUE, step=1)
    values = f(scores, MAX_VALUE)

    for score, value in zip(scores, values):
        print(f'{score}: {value}')

    plt.plot(scores, values)
    plt.show()
