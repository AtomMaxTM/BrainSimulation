import numpy as np

def dot(a, b):
    return np.dot(a, b)

def distance(a, b):
    return round(np.linalg.norm(a - b), 4)

def cords(max_len):
    return np.array([round(np.random.random()*max_len, 4) for _ in range(3)])

def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


# def connect_neurons_tree_unimplemented(self):
#     center = sample(range(self.n_neurons), 1)
#     connect = []
#     for i in center:
#         for j in range(self.n_neurons):
#             if j == i:
#                 continue
#             connect.append((distance(self.neurons[i].cords, self.neurons[j].cords), j))
#     connect = list(map(lambda x: self.neurons[x[1]], nsmallest(self.max_axon, connect, key=lambda x: x[0])))

# def plot_connections_temp(self):
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         t = self.base_neuron.cords
#         ax.scatter(t[0], t[1], t[2], marker='o', c="red")
#         for n in self.connect:
#             temp = n.cords
#             ax.scatter(temp[0], temp[1], temp[2], marker='^', c='green')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         plt.show()
