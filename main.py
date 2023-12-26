import numpy as np
from constants import Constants
from two_layer_perceptron import TwoLayerPerceptron
from draw_plot import draw_approximated_function_approximation_plot


def approximated_function(x):
    return np.sin(x * np.sqrt(Constants.p0() + 1)) + np.cos(x * np.sqrt(Constants.p1() + 1))


if __name__ == '__main__':
    draw_approximated_function_approximation_plot(Constants.lower_bound(), Constants.upper_bound(), approximated_function)
