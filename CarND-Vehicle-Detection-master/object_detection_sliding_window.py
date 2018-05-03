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
    y_start_stop_list = [(400, 600), (400, 600), (500, 750), (500, 750)]
    xy_window_list = [(64, 64), (64, 64), (64, 64), (64, 64)]

    overlapx=0.1
    overlapy=0.5

    for y in y_start_stop_list:
        for xy in xy_window_list:
            windows_list.append(object_detection_utils.slide_window(image, x_start_stop=[None, None],
                                                                    y_start_stop=y, xy_window=xy,
                                                                    xy_overlap=(overlapx, overlapy)))
    window_img = image
    for windows in windows_list:
        color1 = np.random.randint(0, 255)
        color2 = np.random.randint(0, 255)
        color3 = np.random.randint(0, 255)
        window_img = object_detection_utils.draw_boxes(window_img, windows, color=(color1,color2, color3), thick=6)

    plt.imshow(window_img)


if __name__ == '__main__':
    sliding_window_main()
