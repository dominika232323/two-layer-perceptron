import numpy as np


class TwoLayerPerceptron:
    def __init__(self, hidden, learning_rate):
        self._hidden = hidden
        self._learning_rate = learning_rate

        self._hidden_layer_weights = np.random.uniform((-1), 1, size=(hidden, 2))
        self._output_layer_weights = np.random.uniform((-1), 1, size=(1, hidden + 1))

    @property
    def hidden_size(self):
        return self._hidden

    @property
    def learning_rate(self):
        return self._learning_rate

    def train(self, dataset, epochs):
        for epoch in epochs:
            pass

    def _activation_function(self, x):
        return np.tanh(x)

    def predict(self):
        pass
