"""
Utility functions
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_image(i, predictions_array, true_y, img, class_names=None):
    """Plot one prediction"""
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    pred_y = np.argmax(predictions_array)
    if pred_y == true_y:
        color = 'blue'
    else:
        color = 'red'

    true_label = true_y
    pred_label = pred_y
    if class_names:
      true_label = f'{true_y}/{class_names[true_y]}'
      pred_label = f'{pred_y}/{class_names[pred_y]}'

    plt.xlabel(
        f'{pred_label}: '
        f'{100 * np.max(predictions_array):2.0f}% '
        f'(True: {true_label})',
        color=color
    )

def plot_value_array(i, predictions_array, true_label):
    """Plot score for each label for one prediction"""
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')
