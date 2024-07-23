import timeit
from random import random, seed
from time import sleep

import numpy as np

from neuron import Net, Brain, Neuron

neuron_function = lambda x: Neuron(x, 0 if random() > 0.5 else 1, 0 if random() > 0.9 else 1)

# net.add_input(np.array([
#     [1, 1, 1, 1, 1],
#     [0, 0, 1, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 1, 0, 0]
# ]))


input_data = [np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]) for _ in range(5)]
target_data = [np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]) for _ in range(5)]


def main():
    # seed(30)
    net = Net(100, (10,)*2, (5, 5), (5, 5))
    net.create_network(neuron_function, distance_mode=True, verbose=True)
    net.fit(10, 5, input_data, target_data, clear_signals=True)
    net.add_input(np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]))
    net.propagate(5)
    print(net.out.reshape(5, 5))
    net.save('train_test')


if __name__ == '__main__':
    t_0 = timeit.default_timer()
    main()
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    print(f"Elapsed time: {elapsed_time} µs")
    print(f"Elapsed time: {elapsed_time//1000000} s")
