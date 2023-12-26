import matplotlib.pyplot as plt
import numpy as np


def draw_approximated_function_approximation_plot(lower_bound, upper_bound, approximated_func):
    coordinate_system = np.linspace(lower_bound, upper_bound, 200)

    y = approximated_func(coordinate_system)
    plt.plot(coordinate_system, y, label='Funkcja aproksymowana')

    plt.title('Wykres')
    plt.legend()

    plt.show()
    