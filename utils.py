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

def loss(a, b):
    return a - b

def overall_loss(a, b):
    return np.mean((a - b) ** 2)

class SimpleOptimizer:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        pass

    def __call__(self, grad):
        return -(grad * self.lr)


optimizers = {
   'simple': SimpleOptimizer
}


class Optimizer:
    def __init__(self, optimizer='simple', **kwargs):
        self.optimizer = optimizers[optimizer](**kwargs)

    def step(self):
        self.optimizer.step()

    def __call__(self, grad):
        # print(grad)
        return self.optimizer(grad)





