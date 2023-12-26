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
                y = self._output_layer_output(hidden_layer_output)
                error = self._error_function(y_expected, y)

        return error

    def _hidden_layer_output(self, x):
        sumed_array = np.dot(self._hidden_layer_weights, np.vstack((x, 1)))
        return self._activation_function(sumed_array)

    def _output_layer_output(self, hidden_layer_output):
        sumed = np.dot(self._output_layer_weights, np.vstack((hidden_layer_output, 1.0)))
        return np.take(self._activation_function(sumed), 0)

    def _activation_function(self, x):
        return np.tanh(x)

    def _error_function(self, y_expected, y_result):
        return pow(y_result - y_expected, 2)

    def _error_function_derivative(self, y_expected, y_result):
        return (y_result - y_expected) * 2

    def predict(self):
        pass
