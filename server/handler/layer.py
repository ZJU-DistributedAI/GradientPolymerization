import numpy as np


class Layer(object):
    def simple_forward(self, x):
        pass

    def forward(self, x):
        pass

    def backward(self, learning_rate, y, x, *arg):
        pass

    def explore(self, settings):
        pass

    def mutate(self, settings):
        pass

    @staticmethod
    def xavier_activation(dimension):
        boundary = np.sqrt(6. / sum(list(dimension)))
        return np.random.uniform(-boundary, boundary, dimension)

    @staticmethod
    def mutate_array(weight, mutation_probability):
        shape = np.shape(weight)
        mutation = np.random.choice([-1, 1], size=shape, p=[mutation_probability, 1 - mutation_probability])
        return np.multiply(weight, mutation)

