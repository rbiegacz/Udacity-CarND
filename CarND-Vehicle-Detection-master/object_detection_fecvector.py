"""
    this module shows hot to convert an image to 32x32x3
    and still get reasonable feature vector for object detection
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import object_detection_utils


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """
    # Define a function to compute color histogram features
    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.
    # KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
    # IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
    # cv2.imread() INSTEAD YOU START WITH BGR COLOR!
       # Define a function to compute color histogram features
    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.

    :param img:
    :param color_space:
    :param size:
    :return:
    """
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


def main_fecvector():
    """
    TODO: write a description
    :return:
    """
    # Read in an image
    # You can also read cutout2, 3, 4 etc. to see other examples
    image = mpimg.imread('util_images/cutout1.jpg')
    feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')

if __name__ == '__main__':
    main_fecvector()
