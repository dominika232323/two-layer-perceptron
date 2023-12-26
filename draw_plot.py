import matplotlib.pyplot as plt
import numpy as np


def draw_approximated_function_approximation_plot(lower_bound, upper_bound, approximated_func, perceptron):
    coordinate_system = np.linspace(lower_bound, upper_bound, 200)

    y = approximated_func(coordinate_system)
    approximation = perceptron.predict(coordinate_system)

    plt.plot(coordinate_system, y, label='Funkcja aproksymowana')
    plt.plot(coordinate_system, approximation, label='Aproksymacja funkcji')

    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.title('Wykres')
    plt.legend()

    plt.show()
