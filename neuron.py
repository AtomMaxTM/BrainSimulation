import pickle
import timeit
from copy import deepcopy
from heapq import nlargest, nsmallest
from random import sample, random
from time import sleep
from uuid import uuid4
import numpy as np
import vispy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from vispy import app, visuals, gloo
from vispy.scene import visuals
from vispy.util.transforms import perspective, translate, rotate
from utils import flatten_list, cords, distance
from threading import Thread


class Neuron:
    def __init__(self, id, suppress=0, long_shot=0):
        self.id = id
        self.cords = cords(10)
        self.axon_ids = []
        self.dendrite_ids = []
        self.threshold = 1
        self.charge = 0
        self.type = 0
        self.long_shot = long_shot
        self.suppress = suppress
        self.color = 'black' # if type == 0 else 'black'
        self.out = None

    def init_weights(self):
        self.w_dendrites = {i: round(float(j), 6) for i, j in zip(self.dendrite_ids, np.random.randn(len(self.dendrite_ids)))}
        self.w_axons = {i: round(float(j), 6) for i, j in zip(self.axon_ids, np.random.randn(len(self.axon_ids)))}
        if self.suppress:
            for j in self.w_axons.keys():
                self.w_axons[j] = abs(self.w_axons[j])
            for j in self.w_dendrites.keys():
                self.w_dendrites[j] = abs(self.w_dendrites[j])

    def __repr__(self):
        return f'Neuron(id={self.id}, axon_ids={self.axon_ids}, dendrite_ids={self.dendrite_ids})'


class InputNeuron(Neuron):
    def __init__(self, id):
        super().__init__(id)
        self.color = 'green'
        self.type = 1
        self.cords = cords(-4)

    def init_weights(self):
        self.w_axons = {i: round(float(j), 6) for i, j in zip(self.axon_ids, np.random.randn(len(self.axon_ids)))}
        if self.suppress:
            for j in self.w_axons.keys():
                self.w_axons[j] = abs(self.w_axons[j])

    def __repr__(self):
        return f'InputNeuron(id={self.id}, axon_ids={self.axon_ids})'


class OutputNeuron(Neuron):
    def __init__(self, id):
        super().__init__(id)
        self.color = 'red'
        self.type = 2
        self.cords = cords(4) + 10

    def init_weights(self):
        self.w_dendrites = {i: round(float(j), 6) for i, j in zip(self.dendrite_ids, np.random.randn(len(self.dendrite_ids)))}
        if self.suppress:
            for j in self.w_dendrites.keys():
                self.w_dendrites[j] = abs(self.w_dendrites[j])

    def __repr__(self):
        return f'OutputNeuron(id={self.id}, axon_ids={self.axon_ids})'


class Net:
    def __init__(self, n_neurons=None, synapses_count=(None, None), in_size: tuple=tuple(), out_size: tuple=tuple()):
        self.n_neurons = n_neurons
        self.max_axon, self.max_dendrite = synapses_count
        self.in_size = in_size
        self.out_size = out_size
        self.neurons = dict()
        self.in_neurons = []
        self.out_neurons = []
        self.signals = []
        self.merges = 0
        self.out = None

    def create_network(self, neuron_function=lambda x: Neuron(x), *, distance_mode=False, verbose=False):
        self.generate_neurons(neuron_function)
        self.connect_neurons(distance_mode, verbose=verbose)
        for i in self.neurons.values():
            i.init_weights()

    def generate_neurons(self, neuron_function):
        if len(self.in_size):
            s = 1
            for i in self.in_size:
                s *= i
            for _ in range(s):
                id = uuid4().time
                self.neurons[id] = InputNeuron(id)
                self.in_neurons.append(id)
        if len(self.out_size):
            s = 1
            for i in self.out_size:
                s *= i
            for j in range(s):
                id = uuid4().time
                self.neurons[id] = OutputNeuron(id)
                self.out_neurons.append(id)
            self.out = np.zeros(s)
        for _ in range(self.n_neurons):
            id = uuid4().time
            self.neurons[id] = neuron_function(id)

    def connect_neurons(self, distance_mode=False, verbose=False):
        n_ids = [i for i in self.neurons.keys() if i not in self.in_neurons and i not in self.out_neurons]
        saved_n_ids = deepcopy(n_ids)
        current_step = sample(n_ids, self.max_axon)

        if verbose:
            print('| Neurons amount | Max axons | Distance Mode |')
            print("|", str(self.n_neurons).center(14, ' '), "|", str(self.max_axon).center(9, ' '), "|", str(distance_mode).center(13, ' '), "|")
            print('-'*46)
            print('Connecting neurons...')
            print("| Iteration | Neurons Left | Next Step Len |")
            print('-'*44)
            c = 0
            t_0 = timeit.default_timer()
        while (len(n_ids) >= self.max_axon) and ((len(n_ids) != len(current_step)) or len(n_ids) > 20) and len(current_step) > 1:
            if not distance_mode:
                next_step = [sample(n_ids, self.max_axon) for _ in range(len(current_step))]
            else:
                next_step = []
                for i in current_step:
                    current_neuron = self.neurons[i]
                    distances = {distance(current_neuron.cords, self.neurons[i].cords): i for i in n_ids if (i not in self.in_neurons) and ((i not in current_neuron.axon_ids) or (current_neuron.id == i))}
                    next_n = nsmallest(self.max_axon, list(distances.keys())) if not current_neuron.long_shot else nlargest(self.max_axon, list(distances.keys()))
                    next_step.append([distances[i] for i in next_n])
            if verbose:
                c += 1
                print("|", str(c).center(9, ' '), "|", str(len(n_ids)).center(12, ' '), "|", str(len(next_step)).center(13, ' '), "|")
            for base, new in zip(current_step, next_step):
                base_neuron = self.neurons[base]
                for n in new:
                    # removing fully connected neurons
                    if len(base_neuron.axon_ids) >= self.max_axon:
                        try:
                            n_ids.pop(n_ids.index(base))
                        except ValueError:
                            pass
                        break
                    if len(self.neurons[n].dendrite_ids) >= self.max_dendrite:
                        try:
                            n_ids.pop(n_ids.index(n))
                        except ValueError:
                            pass
                        break
                    # if len(base_neuron.dendrite_ids) >= self.max_dendrite:
                    #     continue
                    if (n in base_neuron.axon_ids) or (base_neuron.id == n):
                        continue
                    # connecting neurons
                    base_neuron.axon_ids.append(self.neurons[n].id)
                    self.neurons[n].dendrite_ids.append(base_neuron.id)
            current_step = [i for i in current_step if self.neurons[i].type != 2]
            current_step = flatten_list(next_step)
            current_step = list(set(current_step) - set(current_step).difference(set(n_ids)))

        if len(self.in_size):
            temp_dendrite_ids = {}
            for n in self.in_neurons:
                if not distance_mode:
                    step = []
                    while len(step) < self.max_axon:
                        x = sample(saved_n_ids, 1)
                        if x not in self.in_neurons:
                            step.append(x)
                else:
                    current_neuron = self.neurons[n]
                    distances = {distance(current_neuron.cords, self.neurons[i].cords): i for i in saved_n_ids if temp_dendrite_ids.get(i, -1) < self.max_dendrite + 3 and ((i not in current_neuron.axon_ids) or (current_neuron.id == i))}
                    next_n = nsmallest(self.max_axon, list(distances.keys())) if not current_neuron.long_shot else nlargest(self.max_axon, list(distances.keys()))
                    step = [distances[i] for i in next_n]
                for s in step:
                    self.neurons[n].axon_ids.append(self.neurons[s].id)
                    self.neurons[s].dendrite_ids.append(self.neurons[n].id)
                    temp_dendrite_ids[s] = temp_dendrite_ids.get(s, 0) + 1

        if len(self.out_size):
            temp_axon_ids = {}
            for n in self.out_neurons:
                if not distance_mode:
                    step = []
                    while len(step) < self.max_dendrite:
                        x = sample(saved_n_ids, 1)
                        if x not in self.out_neurons:
                            step.append(x)
                else:
                    current_neuron = self.neurons[n]
                    distances = {distance(current_neuron.cords, self.neurons[i].cords): i for i in saved_n_ids if temp_axon_ids.get(i, -1) < self.max_dendrite + 3 and ((i not in current_neuron.axon_ids) or (current_neuron.id == i))}
                    next_n = nsmallest(self.max_axon, list(distances.keys())) if not current_neuron.long_shot else nlargest(self.max_axon, list(distances.keys()))
                    step = [distances[i] for i in next_n]
                for s in step:
                    self.neurons[n].dendrite_ids.append(self.neurons[s].id)
                    self.neurons[s].axon_ids.append(self.neurons[n].id)
                    temp_axon_ids[s] = temp_axon_ids.get(s, 0) + 1

        if verbose:
            print('-' * 44)
            print('Network generated successfully')
            print("-" * 44)
            print("|", "Time Elapsed".center(40, " "), "|")
            print("-" * 44)
            t_1 = timeit.default_timer()
            elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
            print("|", "µs".center(18, " "), '|', "s".center(19, " "), "|")
            print("|", f"{elapsed_time}".center(18, " "), '|', f"{elapsed_time//1000000}".center(19, " "), "|")
            print('-' * 44)

    def step(self):
        new_signals = []
        for signal in self.signals:
            l_signal = self.neurons[signal[-1]]
            if l_signal.charge >= l_signal.threshold:
                if not isinstance(l_signal, OutputNeuron):
                    for k, v in l_signal.w_axons.items():
                        n = self.neurons[k]
                        n.charge += v * n.w_dendrites[signal[-1]]
                        new_signals.append(signal + [k])
                    l_signal.charge = 0
                else:
                    self.out[self.out_neurons.index(l_signal.id)] += l_signal.charge

        self.signals = new_signals

        print(f'Signals amount: {len(new_signals)}')

    def merge(self, net, self_2_net_merge_neurons_n, net_2_self_merge_neurons_n, new_axon, new_dendrite):
        new = deepcopy(self)
        temp = deepcopy(net.neurons)
        new.merges += 1
        for i in temp.values():
            i.cords += np.array([10*new.merges, 10*new.merges, 10*new.merges])
        new.neurons.update(temp)
        del temp
        new.n_neurons += net.n_neurons
        new.signals.extend(net.signals)

        ks = list(self.neurons.keys())
        kn = list(net.neurons.keys())
        self2net_neurons = sample(ks, self_2_net_merge_neurons_n)
        net2self_neurons = sample(kn, net_2_self_merge_neurons_n)

        used = []
        for neuron in self2net_neurons:
            for i in range(new_axon):
                neuron_connect = sample(kn, 1)[0]
                while neuron_connect in used:
                    neuron_connect = sample(kn, 1)[0]
                conn_neuron = new.neurons[neuron_connect]
                conn_neuron.dendrite_ids.append(neuron)
                conn_neuron.w_dendrites[neuron] = round(float(np.random.randn()), 6)
                new.neurons[neuron].axon_ids.append(neuron_connect)
                new.neurons[neuron].w_axons[neuron_connect] = round(float(np.random.randn()), 6)
                used.append(neuron_connect)

        used = []
        for neuron in net2self_neurons:
            for i in range(new_dendrite):
                neuron_connect = sample(ks, 1)[0]
                while neuron_connect in used:
                    neuron_connect = sample(ks, 1)[0]
                conn_neuron = new.neurons[neuron_connect]
                conn_neuron.dendrite_ids.append(neuron)
                conn_neuron.w_dendrites[neuron] = round(float(np.random.randn()), 6)
                new.neurons[neuron].axon_ids.append(neuron_connect)
                new.neurons[neuron].w_axons[neuron_connect] = round(float(np.random.randn()), 6)
                used.append(neuron_connect)

        return new

    def set_start_signal_debug(self, neurons):
        for i in neurons:
            n = list(self.neurons.keys())[i]
            self.neurons[n].charge = 10
            self.signals.append([n])

    def add_input(self, data: np.ndarray):
        if type(data) == np.ndarray:
            data = data.flatten()
        else:
            data = flatten_list(data)
        for i, d in enumerate(data.flatten()):
            self.neurons[self.in_neurons[i]].charge = d
            self.signals.append([self.in_neurons[i]])

    def propagate(self, steps, delay=0):
        if not sleep:
            for i in range(steps):
                if len(self.signals) == 0:
                    print(f'Signal died at step {i+1}/{steps}')
                    break
                print(f"Step: {i}", end=" ")
                self.step()
        else:
            for i in range(steps):
                if len(self.signals) == 0:
                    print(f'Signal died at step {i + 1}/{steps}')
                    break
                print(f"Step: {i}", end=" ")
                self.step()
                sleep(delay)
        print(f'Output: {self.out.reshape(self.out_size)}')

    def plot_propagation(self, steps, delay=1):
        self.step()
        def timed_step(net: Net, steps, delay):
            sleep(5)
            for i in range(steps):
                if len(net.signals) == 0:
                    print(f'Signal died at step {i+1}/{steps}')
                    break
                net.step()
                sleep(delay)
            else:
                print(f'Propagated {i+1} steps')

        t = Thread(target=timed_step, args=(self, steps, delay))
        canvas = Plotter(self, animation=True, step_func=t)
        canvas.show()
        app.run()

    def show_network_parallel(self):
        p = Thread(target=visualizator, args=(self,))
        p.start()

    def plot_connections(self):
        x, y, z = [], [], []

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for n in self.neurons.values():
            ax.scatter(n.cords[0], n.cords[1], n.cords[2], c=n.color, marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')


        for n in self.neurons.values():
            x, y, z = n.cords
            for n2 in n.axon_ids:
                x1, y1, z1 = self.neurons[n2].cords
                ax.plot([x, x1], [y, y1], [z, z1], color='b' if n.color != 'green' else 'black')

        plt.show()

    def plot_gpu(self):
        canvas = Plotter(self, )
        canvas.show()
        app.run()

    def animated_plot_connections(self, speed=50):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for n in self.neurons.values():
            ax.scatter(n.cords[0], n.cords[1], n.cords[2], c=n.color, marker='o')
        lines = []

        def update(frame):
            i, j = frame
            x, y, z = i.cords
            x1, y1, z1 = j.cords
            line, = ax.plot([x, x1], [y, y1], [z, z1], color='b' if n.color != 'green' else 'black')

        def frames():
            for i in self.neurons.values():
                for j in i.axon_ids:
                    yield i, self.neurons[j]

        anim = FuncAnimation(fig, update, frames=frames, interval=speed, repeat=False)
        plt.show()

    def save(self, filename='net'):
        with open(filename+'.pkl', 'wb') as file:
            pickle.dump(
                {
                    'neurons': self.neurons,
                    'max_axon': self.max_axon,
                    'max_dendrite': self.max_dendrite,
                    'n_neurons': self.n_neurons,
                    'merges': self.merges,
                    'in_size': self.in_size,
                    'out_size': self.out_size,
                    'in_neurons': self.in_neurons,
                    'out_neurons': self.out_neurons,
                    'signals': self.signals,
                    'out': self.out
                },
                file)

    def load(self, filename='net.pkl'):
        with open(filename+'.pkl', 'rb') as file:
            obj = pickle.load(file)
        self.neurons = obj['neurons']
        self.max_axon = obj['max_axon']
        self.max_dendrite = obj['max_dendrite']
        self.n_neurons = obj['n_neurons']
        self.merges = obj['merges']
        self.in_size = obj['in_size']
        self.out_size = obj['out_size']
        self.in_neurons = obj['in_neurons']
        self.out_neurons = obj['out_neurons']
        self.signals = obj['signals']
        self.out = obj['out']


def visualizator(net: Net):
    canvas = Plotter(net, real_time=True)
    canvas.show()
    app.run()

class Plotter(app.Canvas):
    VERT_SHADER = """
    #version 120
    attribute vec3 a_position;
    attribute vec3 a_color;
    varying vec3 v_color;
    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_projection;
    void main() {
        v_color = a_color;
        gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
        gl_PointSize = 10.0;
    }
    """

    FRAG_SHADER = """
    #version 120
    varying vec3 v_color;
    void main() {
        gl_FragColor = vec4(v_color, 1.0);
    }
    """

    def __init__(self, network: Net, real_time=False, animation=False, step_func: Thread=None):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        self.net = network
        self.animation = animation
        self.real_time = real_time
        self.colors = {
            'blue': [0, 0, 1],
            'yellow': [1, 1, 0],
            'green': [0, 1, 0],
            'red':  [1, 0, 0],
            'black': [0, 0, 0]
        }
        self.program = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        scatter_data = np.array([n.cords for n in self.net.neurons.values()], dtype=np.float32)
        scatter_colors = np.array([self.colors[n.color] for n in self.net.neurons.values()], dtype=np.float32)


        self.scatter_buffer = gloo.VertexBuffer(scatter_data)
        self.color_buffer = gloo.VertexBuffer(scatter_colors)

        self.program['a_position'] = self.scatter_buffer
        self.program['a_color'] = self.color_buffer

        self.lines = []
        if not self.animation and not real_time:
            for n in self.net.neurons.values():
                x, y, z = n.cords
                for n2 in n.axon_ids:
                    x1, y1, z1 = self.net.neurons[n2].cords
                    line_data = np.array([[x, y, z], [x1, y1, z1]], dtype=np.float32)
                    line_colors = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)  # blue color for lines
                    self.lines.append((gloo.VertexBuffer(line_data), gloo.VertexBuffer(line_colors)))

        self.model = np.eye(4, dtype=np.float32)
        self.view = translate((0, 0, -5))
        self.projection = perspective(45.0, self.size[0] / float(self.size[1]), 1.0, 100.0)

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.theta = 0
        self.phi = 0

        gloo.set_state(clear_color='white', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))

        self._init_transforms()
        if animation:
            step_func.start()

    def _init_transforms(self):
        self.translate = [0, 0, -5]
        self.scale = 1
        self.rotation = np.eye(4, dtype=np.float32)

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(45.0, width / float(height), 1.0, 100.0)

    def on_mouse_wheel(self, event):
        self.translate[2] += event.delta[1]

    def on_mouse_move(self, event):
        if event.is_dragging:
            dx = event.pos[0] - event.last_event.pos[0]
            dy = event.pos[1] - event.last_event.pos[1]
            self.theta += dx * 0.5
            self.phi += dy * 0.5
            self.rotation = np.dot(rotate(self.theta, (0, 1, 0)), rotate(self.phi, (1, 0, 0)))
            self.update()

    def on_key_press(self, event):
        if event.key == 'a':
            self.translate[0] += 0.5
        elif event.key == 'd':
            self.translate[0] -= 0.5
        elif event.key == 'w':
            self.translate[1] -= 0.5
        elif event.key == 's':
            self.translate[1] += 0.5
        self.update()

    def on_timer(self, event):
        self.update()

    def on_draw(self, event):
        gloo.clear()

        self.program['u_model'] = np.dot(self.rotation, translate(self.translate))
        self.program['u_view'] = self.view
        self.program['u_projection'] = self.projection

        # Draw points
        self.program['a_position'] = self.scatter_buffer
        self.program['a_color'] = self.color_buffer
        self.program.draw('points')

        if not self.animation and not self.real_time:
            # Draw lines
            for line_buffer, color_buffer in self.lines:
                self.program['a_position'] = line_buffer
                self.program['a_color'] = color_buffer
                self.program.draw('lines')
        elif self.animation or self.real_time:
            for signal in self.net.signals:
                if len(signal) < 2:
                    continue
                x, y, z = self.net.neurons[signal[-2]].cords
                x1, y1, z1 = self.net.neurons[signal[-1]].cords
                line_data = np.array([[x, y, z], [x1, y1, z1]], dtype=np.float32)
                line_colors = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)  # blue color for lines
                self.program['a_position'] = gloo.VertexBuffer(line_data)
                self.program['a_color'] = gloo.VertexBuffer(line_colors)
                self.program.draw('lines')


class Brain:
    def __init__(self):
        self.networks = []

    def add_network(self, n_neurons, synapses_count, neuron_function=None):
        if neuron_function is None:
            neuron_function = lambda x: Neuron(x)
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

