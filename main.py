import timeit
from random import random, seed
from neuron import Net, Brain, Neuron

neuron_function = lambda x: Neuron(x, 1)

def main():
    seed(30)
    net1 = Net(100, (10,)*2)
    net1.create_network(distance_mode=True, verbose=True)
    # net2 = Net(200, (4,)*2)
    # net2.create_network(distance_mode=True, verbose=True)
    # net1 = net1.merge(net2, 10, 10, 4, 4)
    # net1.load('1000_10')
    # net1.plot_gpu()
    net1.set_start_signal(list(range(20)))
    # net1.propagate(10)
    # net1.set_start_signal([1, 10, 21, 20, 30, 40, 50])
    # net1.plot_propagation(25, 0.5)
    net1.show_network_parallel()
    net1.propagate(20, delay=0.25)

if __name__ == '__main__':
    t_0 = timeit.default_timer()
    main()
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    print(f"Elapsed time: {elapsed_time} µs")
    print(f"Elapsed time: {elapsed_time//1000000} s")
