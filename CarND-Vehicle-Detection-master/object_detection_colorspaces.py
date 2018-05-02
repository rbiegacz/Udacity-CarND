"""
This module shows how to use different color spaces to identify clusters of pixels;
Clusters of pixels are useful while discoverying objects like sky, road, vehicle, etc.

Use this to first explore some video frames, and see if you can locate clusters of colors
that correspond to the sky, trees, specific cars, etc.
Here are some sample images for you to use

A lot of code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

AXIS_LIMITS = ((0, 255), (0, 255), (0, 255))


def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=AXIS_LIMITS):
    """
    Plot pixels in 3D.
    :param pixels:
    :param colors_rgb:
    :param axis_labels:
    :param axis_limits:
    :return:
    """

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def main_color_spaces():
    """
    transforming an image to a different color space
    :return: nothing; if everything is correct than a picture will be drawn
    """
    # Read a color image
    img = cv2.imread('util_images/cutout1.jpg')

    # Select a small fraction of pixels to plot by sub-sampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

    # Convert sub-sampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()

    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()


if __name__ == '__main__':
    main_color_spaces()
