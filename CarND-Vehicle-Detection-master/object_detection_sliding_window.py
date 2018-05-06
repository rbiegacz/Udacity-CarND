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
    image = mpimg.imread('util_images/bbox-example-image.png')
    windows_list = []
    x_start_stop_list = [[50, 400], [400, 750], [750, 1200]]
    y_start_stop_list = [(420, 700), (400, 600), (420, 700)]
    xy_window_list = [(128, 128), (64, 64), (128, 128)]
    overlapx=0.75
    overlapy=0.75

    for y_start_stop, xy, x_start_stop in zip(y_start_stop_list,xy_window_list,x_start_stop_list):
        windows_list += object_detection_utils.slide_window(image, x_start_stop=x_start_stop,
                                                           y_start_stop=y_start_stop, xy_window=xy,
                                                           xy_overlap=(overlapx, overlapy))
    window_img = object_detection_utils.draw_boxes(image, windows_list, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)


if __name__ == '__main__':
    sliding_window_main()
