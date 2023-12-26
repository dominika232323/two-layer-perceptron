import numpy as np
from sklearn.utils import shuffle


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
        for i in range(1, epochs + 1):
            shuffled_dataset = shuffle(dataset, random_state=i)

            for point in shuffled_dataset:
                x = point[0]
                y_expected = point[1]

                hidden_layer_output = self._hidden_layer_output(x)

        return hidden_layer_output

    def _hidden_layer_output(self, x):
        sums_array = np.dot(self._hidden_layer_weights, np.vstack((x, 1)))
        return self._activation_function(sums_array)

    def _activation_function(self, x):
        return np.tanh(x)

    def predict(self):
        pass
