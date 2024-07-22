import timeit
from random import random, seed
from time import sleep

import numpy as np

from neuron import Net, Brain, Neuron, InputNeuron, OutputNeuron

neuron_function = lambda x: Neuron(x, 1)

def main():
    # seed(30)
    net = Net(100, (10,)*2, (5, 5), (5, 5))
    # net.create_network(lambda x: Neuron(x, 0 if random() > 0.5 else 1, 0 if random() > 0.9 else 1), distance_mode=True, verbose=True)
    # net.save('test')
    net.load('test')
    # net.plot_gpu()
    # net.add_input(np.random.randn(5, 5))
    net.add_input(np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]))
    # net.show_network_parallel()
    # sleep(3)
    net.propagate(10)


if __name__ == '__main__':
    t_0 = timeit.default_timer()
    main()
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    print(f"Elapsed time: {elapsed_time} µs")
    print(f"Elapsed time: {elapsed_time//1000000} s")
