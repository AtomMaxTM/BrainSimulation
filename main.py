import timeit
from random import random
from neuron import Net, Brain, Neuron

neuron_function = lambda x: Neuron(x, 1)

def main():
    # brain = Brain()
    # brain.add_network(1000, (2,)*2,)
    # print('checkpoint')
    # brain.plot_connections_vispy()
    # brain.plot_fast()
    # brain.set_start_signal([0, 1])
    # net = Net(1000, (2,)*2)
    # net.create_network()
    # net.plot_connections()
    # print('checkpoint')
    # net2 = Net(1000, (2,)*2)
    # net2.create_network(distance_mode=True, no_init=True)
    # net2.plot_connections()
    # net3 = net.merge(net2, 10, 10, 2, 2)
    # net3.save('merged')
    # net3.plot_connections()
    # net.load('1000.pkl')
    # net.save('1000.pkl')
    # net.set_start_signal([87, 345, 543, 777])
    # net.propagate(3)
    #   --------------------------------

    net1 = Net(1000, (4,)*2)
    # net1.create_network(distance_mode=True, verbose=True)
    net1.load('1000')
    net1.plot_gpu()

if __name__ == '__main__':
    t_0 = timeit.default_timer()
    main()
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    print(f"Elapsed time: {elapsed_time} µs")
    print(f"Elapsed time: {elapsed_time//1000000} s")

# 3