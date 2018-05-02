"""
    This module explore datasets of car and non-cars.
    A lot of code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import object_detection_utils


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    """
    TODO: write a function description here
    :param car_list:
    :param notcar_list:
    :return:
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    print(car_list[0])
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


def main_look_for_car():
    """
    TODO: write a function description
    function showing how to dig out car pictures
    :return:
    """
    # from skimage.feature import hog
    # from skimage import color, exposure
    # images are divided up into vehicles and non-vehicles

    notcars = glob.glob('util_images/non-vehicles_smallset/*/*.jpeg')
    cars = glob.glob('util_images/vehicles_smallset/*/*.jpeg')

    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # Plot the examples
    _ = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')

if __name__ == '__main__':
    main_look_for_car()
