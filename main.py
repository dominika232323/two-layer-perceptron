import numpy as np
from constants import Constants
from two_layer_perceptron import TwoLayerPerceptron
from draw_plot import draw_approximated_function_approximation_plot


def approximated_function(x):
    return np.sin(x * np.sqrt(Constants.p0() + 1)) + np.cos(x * np.sqrt(Constants.p1() + 1))


def create_points_for_function(function, lower_bound, upper_bound, how_many):
    points = []
    x_coords = np.linspace(lower_bound, upper_bound, how_many)

    for x in x_coords:
        y = function(x)
        points.append([x, y])

    return points


if __name__ == '__main__':
    dataset = create_points_for_function(approximated_function, Constants.lower_bound(), Constants.upper_bound(), 100)
    perceptron = TwoLayerPerceptron(7, 0.003)
    draw_approximated_function_approximation_plot(Constants.lower_bound(), Constants.upper_bound(), approximated_function, perceptron)
