"""
    this module contains utility functions for object detection
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def util_color_hist(img, bins_number=32, bins_range=(0, 256)):
    """
    Define a function to compute color histogram features
    :param img:
    :param bins_number:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=bins_number, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=bins_number, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=bins_number, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


def main_histogram():
    """
    This function that shows how histograms over red, green and blue colors
    might be helpful in object detection
    :return:
    """
    image = mpimg.imread('util_images/cutout1.jpg')
    rh, gh, bh, bin_centers, _ = util_color_hist(image, bins_number=32, bins_range=(0, 256))

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bin_centers, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bin_centers, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bin_centers, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
    else:
        print('Your function is returning None for at least one variable...')


if __name__ == '__main__':
    main_histogram()
