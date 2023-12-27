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

                hidden_layer_output, output = self._forward(x)
                self._backpropagation(hidden_layer_output, output, y_expected, x)

    def _forward(self, x):
        hidden_output = self._hidden_layer_output(x)
        output = self._output_layer_output(hidden_output)
        return hidden_output, output

    def _hidden_layer_output(self, x):
        sumed_array = np.dot(self._hidden_layer_weights, np.vstack((x, 1)))
        return self._activation_function(sumed_array)

    def _output_layer_output(self, hidden_layer_output):
        sumed = np.dot(self._output_layer_weights, np.vstack((hidden_layer_output, 1.0)))
        return np.take(self._activation_function(sumed), 0)

    def _backpropagation(self, hidden_layer_output, output, y_expected, x):
        delta = self._error_function_derivative(y_expected, output) * self._activation_function_derivative(output)

        new_array = np.vstack((hidden_layer_output, 1)).T
        output_layer_weights_diff = new_array * delta
        hidden_layer_weights_diff = delta * self._activation_function_derivative(hidden_layer_output) * x

        self._output_layer_weights -= self._learning_rate * output_layer_weights_diff
        self._hidden_layer_weights -= self._learning_rate * hidden_layer_weights_diff

        # output_gradient = self._error_function_derivative(y_expected, output) * np.vstack((hidden_layer_output, 1.0)).T
        #
        # diff_matrix = np.ones((self._hidden, 1), dtype = float) - np.square(hidden_layer_output)
        # hidden_gradient = np.dot(self._error_function_derivative(y_expected, output)*(self._output_layer_weights.T)[:-1, :]*diff_matrix, np.array([[x, 1]]))
        #
        # self._output_layer_weights -= self._learning_rate * output_gradient
        # self._hidden_layer_weights -= self._learning_rate * hidden_gradient

    def _activation_function(self, x):
        return np.tanh(x)

    def _activation_function_derivative(self, x):
        return 1 - pow(np.tanh(x), 2)

    def _error_function(self, y_expected, y_result):
        return pow(y_result - y_expected, 2)

    def _error_function_derivative(self, y_expected, y_result):
        return (y_result - y_expected) * 2

    def predict(self, x_dataset):
        results = []

        for x in x_dataset:
            h, y = self._forward(x)
            results.append(y)

        return results
