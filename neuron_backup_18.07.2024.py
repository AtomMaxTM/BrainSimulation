import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import flatten_list, cords
from random import sample
import vispy
from vispy.scene import visuals


class Neuron:
    def __init__(self, id):
        self.id = id
        self.cords = cords(10)
        self.axon_ids = []
        self.dendrite_ids = []
        self.threshold = 0.5
        self.charge = 0

    def init_weights(self):
        self.w_dendrites = {i: round(float(j), 6) for i, j in zip(self.dendrite_ids, np.random.randn(len(self.dendrite_ids)))}
        self.w_axons = {i: round(float(j), 6) for i, j in zip(self.axon_ids, np.random.randn(len(self.axon_ids)))}

    def __repr__(self):
        return f'Neuron(id={self.id}, axon_ids={self.axon_ids}, dendrite_ids={self.dendrite_ids})'


class Net:
    def __init__(self, n_neurons, synapses_count):
        self.n_neurons = n_neurons
        self.max_axon, self.max_dendrite = synapses_count
        self.neurons = []
        self.signals = []
        self.in_connectors: list[Connector] = []
        self.out_connectors: list[Connector] = []

    def create_network(self, neuron_function):
        self.__generate_neurons(neuron_function)
        self.__connect_neurons()
        for i in self.neurons:
            i.init_weights()

    def __generate_neurons(self, neuron_function):
        self.neurons = [neuron_function(i) for i in range(self.n_neurons)]

    def __connect_neurons(self):
        n_ids = list(range(self.n_neurons))
        current_step = sample(range(self.n_neurons), self.max_axon)
        # c = 0
        # print("Iteration | Neurons Left | Next Step Len | Current Step Len")
        while len(n_ids) > self.max_axon:
            # c += 1
            next_step = [sample(n_ids, self.max_axon) for _ in range(len(current_step))]
            # print(c, "|", len(n_ids), "|", len(next_step), "|", len(current_step))
            for base, new in zip(current_step, next_step):
                base_neuron = self.neurons[base]
                for n in new:
                    # removing fully connected neurons
                    if len(base_neuron.dendrite_ids) == self.max_dendrite:
                        continue
                    if len(base_neuron.axon_ids) == self.max_axon:
                        try:
                            n_ids.pop(n_ids.index(base))
                        except ValueError:
                            pass
                        break
                    if len(self.neurons[n].dendrite_ids) == self.max_dendrite:
                        try:
                            n_ids.pop(n_ids.index(n))
                        except ValueError:
                            pass
                        break
                    if (n in base_neuron.axon_ids):
                        continue
                    if base_neuron.id == n:
                        continue
                    # connecting neurons
                    base_neuron.axon_ids.append(self.neurons[n].id)
                    self.neurons[n].dendrite_ids.append(base_neuron.id)
            current_step = flatten_list(next_step)
            current_step = list(set(current_step) - set(current_step).difference(set(n_ids)))

    def step(self):
        new_signals = []

        for c in self.in_connectors:
            for s in c.in_signals:
                new_signals.append(s)

        for signal in self.signals:
            if signal[-1] in :
            l_signal = self.neurons[signal[-1]]
            if l_signal.charge >= l_signal.threshold:
                for k, v in l_signal.w_axons.items():
                    n = self.neurons[k]
                    n.charge += v * n.w_dendrites[signal[-1]]
                    new_signals.append(signal + [k])
                l_signal.charge = 0
        self.signals = new_signals


    # Коннектор добавляется в списки коннекторов у объектов и на вызове step() чекаются входящие сигналы,
    # а также проверяются нейроны которые должны перекинуть сигнал на другую сеть

    def plot_connections(self):
        x, y, z = [], [], []

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for n in self.neurons:
            ax.scatter(n.cords[0], n.cords[1], n.cords[2], c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')


        for n in self.neurons:
            x, y, z = n.cords
            for n2 in n.axon_ids:
                x1, y1, z1 = self.neurons[n2].cords
                ax.plot([x, x1], [y, y1], [z, z1], color='b')

        plt.show()

    def animated_plot_connections(self, speed=50):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [neuron.cords[0] for neuron in self.neurons]
        y = [neuron.cords[1] for neuron in self.neurons]
        z = [neuron.cords[2] for neuron in self.neurons]
        ax.scatter(x, y, z, c='r', marker='o')
        lines = []

        def update(frame):
            i, j = frame
            x, y, z = i.cords
            x1, y1, z1 = j.cords
            line, = ax.plot([x, x1], [y, y1], [z, z1], color='b')

        def frames():
            for i in self.neurons:
                for j in i.axon_ids:
                    yield i, self.neurons[j]

        anim = FuncAnimation(fig, update, frames=frames, interval=speed, repeat=False)
        plt.show()

    def save(self, filename='net.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(
                {
                    'neurons': self.neurons,
                    'max_axon': self.max_axon,
                    'max_dendrite': self.max_dendrite,
                    'n_neurons': self.n_neurons
                },
                file)

    def load(self, filename='net.pkl'):
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        self.neurons = obj['neurons']
        self.max_axon = obj['max_axon']
        self.max_dendrite = obj['max_dendrite']
        self.n_neurons = obj['n_neurons']

class Switch:
    def __init__(self, connector):


class Connector:
    def __init__(self, in_net: Net, out_net: Net, n_conn_in: int, n_conn_out: int, n_axons: int):
        self.in_signals = []
        self.out_signals = []

        self.in_neurons = []
        self.out_neurons = []

        self.in_net = in_net
        self.out_net = out_net
        in_net.out_connectors.append(self)
        out_net.in_connectors.append(self)

    def put_in(self, signal):
        self.in_signals.append(signal)

    def put_out(self, signal):
        self.out_signals.append(signal)


class Brain:
    def __init__(self):
        self.networks = []

    def add_network(self, n_neurons, synapses_count, neuron_function=None):
        if neuron_function is None:
            neuron_function = lambda x: Neuron(x)
        multiplier = 0 + sum(i.n_neurons for i in self.networks)
        neuron_function = lambda x: neuron_function(x+multiplier)
        net = Net(n_neurons, synapses_count)
        net.create_network(neuron_function)
        self.networks.append(net)

    def propagate(self, steps):
        for _ in range(steps):
            self.step()

    def step(self):
        for net in self.networks:
            net.step()

    def set_start_signal(self, net_idx, neurons):
        for i in neurons:
            self.networks[net_idx].signals.append([i])

    def plot(self, animate=False, speed=50):
        if not animate:
            self.networks[0].plot_connections()
        else:
            self.networks[0].animated_plot_connections(speed=speed)

    def plot_vispy(self):
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'

        scatter_data = np.array([n.cords for n in self.networks[0].neurons])
        scatter = vispy.scene.visuals.Markers()
        scatter.set_data(scatter_data, face_color='red', size=10)
        view.add(scatter)

        for n in self.networks[0].neurons:
            x, y, z = n.cords
            for n2 in n.axon_ids:
                x1, y1, z1 = self.networks[0].neurons[n2].cords
                line_data = np.array([[x, y, z], [x1, y1, z1]])
                line = vispy.scene.visuals.Line(line_data, color='blue', width=2)
                view.add(line)

        # axis = vispy.scene.visuals.XYZAxis(parent=view.scene)

        canvas.app.run()

    def plot_neurons_vispy(self):
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = canvas.central_widget.add_view()

        x = np.array([neuron.cords[0] for neuron in self.networks[0].neurons])
        y = np.array([neuron.cords[1] for neuron in self.networks[0].neurons])
        z = np.array([neuron.cords[2] for neuron in self.networks[0].neurons])

        scatter = visuals.Markers()
        scatter.set_data(np.c_[x, y, z], face_color='red', size=10)
        view.add(scatter)

        view.camera = 'arcball'  # or try 'arcball'

        lines = []

        def update(event):
            nonlocal lines
            # for line in lines:
            #     view.remove(line)
            # lines = []

            for signal in self.networks[0].signals:
                start_cord = self.networks[0].neurons[signal[-2]].cords
                end_cord = self.networks[0].neurons[signal[-1]].cords
                line_data = np.array([start_cord, end_cord])
                line = visuals.Line()
                line.set_data(line_data, color='blue')
                view.add(line)
                lines.append(line)

            canvas.update()

        timer = vispy.app.Timer(interval=0.5, connect=update, start=True)
        vispy.app.run()

