"""
    This module implements a mechanism of sliding windows;
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import object_detection_utils

def sliding_window_main():
    """
    This function demonstrates how to calcualte sliding windows for a given image.
    :return: it doesn't return anything if everything works perfectly
    """
    image = mpimg.imread('util_images/bbox-example-image.jpg')
    windows = object_detection_utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = object_detection_utils.draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)


if __name__ == '__main__':
    sliding_window_main()
