"""
    This module contains functions for HOG feature extraction
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""
import glob
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import object_detection_utils

def print_parameters(color_space, orient, pix_per_cell, cell_per_block, hog_channel):
    """
    This function is used only to display information. It formats print output for arguments delivered.
    :param color_space: color space
    :param orient:
    :param pix_per_cell:
    :param cell_per_block: number of cells per block
    :param hog_channel: HOG channel used
    :return:
    """
    print("Color Space: {:6.6}, Hog Channel: {}, Orient: {}, Pix per Cell: {:2}, Cell per Block: {}: calculation".format(color_space, hog_channel, orient, pix_per_cell, cell_per_block))


def train_model(model_params):
    """
    This function trains linear SVC model based on the passed parameters
    :param model_params: parameters of the model
    :return: trained linearSVC model
    """
    cars = glob.glob('util_images/vehicles/**/*.png')
    not_cars = glob.glob('util_images/non-vehicles/**/*.png')
    sample_size = model_params["sample_size"]
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    color_item = model_params["color_item"]
    orient = model_params["orient"]
    pix_per_cell = model_params["pix_per_cell"]
    cells_per_block = 2
    hog_channels = model_params["hog_channel"]


    print_parameters(color_item, orient, pix_per_cell, cells_per_block, hog_channels)

    car_features = object_detection_utils.extract_features(cars, color_space=color_item, orient=orient,
                                                           pix_per_cell=pix_per_cell, cell_per_block=cells_per_block,
                                                           hog_channel=hog_channels)
    notcar_features = object_detection_utils.extract_features(not_cars, color_space=color_item, orient=orient,
                                                              pix_per_cell=pix_per_cell, cell_per_block=cells_per_block,
                                                              hog_channel=hog_channels)
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    svc.fit(X_train, y_train)
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    model = dict()
    model["linearSVC"] = svc
    model["X_scaler"] = X_scaler
    model["color_item"] = model_params["color_item"]
    model["orient"] = model_params["orient"]
    model["pix_per_cell"] = model_params["pix_per_cell"]
    model["cells_per_block"] = 2
    model["hog_channel"] = model_params["hog_channel"]
    model["sample_size"] = model_params["sample_size"]
    model["spatial_feat"] = True
    model["hist_feat"] = True
    model["hog_feat"] = True
    return model

def main_hog():
    """
    TODO: correct description of this function
    function showing how feature extraction works
    :return:
    """

    cars = glob.glob('util_images/vehicles/**/*.png')
    not_cars = glob.glob('util_images/non-vehicles/**/*.png')

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 100
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    experiments_dict = dict()
    color_space = {'RGB', 'HSV', 'HLS', 'YCrCb', 'YUV'} # {'LUV'}
    orients = [8, 9, 10, 11]
    pixels_per_block = [8, 16]
    cells_per_block = [2]
    hog_channels = ["ALL"] #[0, 1, 2, "ALL"]
    for color_item in color_space:
        for orient in orients:
            for pix_per_cell in pixels_per_block:
                for cell_per_block in cells_per_block:
                    for hog_channel in hog_channels:
                        print_parameters(color_item, orient, pix_per_cell, cell_per_block, hog_channel)
                        t = time.time()
                        car_features = object_detection_utils.extract_features(cars, color_space=color_item, orient=orient,
                                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                        hog_channel=hog_channel)
                        notcar_features = object_detection_utils.extract_features(not_cars, color_space=color_item, orient=orient,
                                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                           hog_channel=hog_channel)
                        t2 = time.time()
                        time_extract_hog_features = round(t2 - t, 2)

                        # Create an array stack of feature vectors
                        X = np.vstack((car_features, notcar_features)).astype(np.float64)

                        # Define the labels vector
                        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

                        # Split up data into randomized training and test sets
                        rand_state = np.random.randint(0, 100)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

                        # Fit a per-column scaler
                        X_scaler = StandardScaler().fit(X_train)
                        # Apply the scaler to X
                        X_train = X_scaler.transform(X_train)
                        X_test = X_scaler.transform(X_test)

                        # Use a linear SVC
                        svc = LinearSVC()
                        # Check the training time for the SVC
                        t = time.time()
                        svc.fit(X_train, y_train)
                        t2 = time.time()
                        # Check the score of the SVC
                        accuracy = round(svc.score(X_test, y_test), 4)
                        # Check the prediction time for a single sample
                        experiments_dict["{}_{}_{}_{}_{}".format(color_item, hog_channel, orient, pix_per_cell, cell_per_block)] = \
                            {"color_item": color_item, "hog_channel": hog_channel,
                             "orient": orient, "pix_per_cell":pix_per_cell,
                             "cell_per_block": cell_per_block,
                             "time_extract_hog_features": time_extract_hog_features,
                             "accuracy": accuracy,
                             "feature vector length": len(X_train[0]),
                             "linearSVC": svc,
                             "X_scaler": X_scaler,
                             "spatial_size": (32, 32),
                             "hist_bins": 32}
    return experiments_dict


def print_results(experiments):
    """
    TODO: deliver description of this function
    :param experiments:
    :return:
    """
    print("CS means Color Space")
    print("Or means Orientation")
    print("C/B means Cells per Block")
    print("P/C means Pixels per Cell")
    print("\n")
    # T(pred):{}\t| T(ext):{}\t|
    for key in experiments:
        results = experiments[key]
        print("| CS:{}\t| Or:{:02}\t| C/B:{}\t| P/C:{:02}\t| Acc {:06.4f}\t| F:{:06}\t|"\
              .format(results["color_item"], results["orient"],
                      results["cell_per_block"], results["pix_per_cell"], results["accuracy"],
                      # results["time_for_prediction"], results["time_extract_hog_features"],
                      results["feature vector length"]))


def get_models(experiments, accuracy=0.99):
    """
    Return only models that have accuracy specified by 'accuracy' or higher
    :param experiments: dictionary of all experiments
    :param accuracy: specified the accuracy we are looking for
    :return: dictionary of experiments matching the criteria
    """
    matching_experiments = dict()
    for key in experiments:
        if experiments[key]['accuracy'] > accuracy:
            matching_experiments[key] = experiments[key]
    return matching_experiments


def get_best_model(experiments, model_type='HLS'):
    """
    This function searches thru all the elements of experiments dictionary and retrieves the best model,
    i.e. model with the best accuracy.
    :param experiments: dictionary of all models created for vehicle recognition
    :return: the entry for the best model
    """
    best_model_accuracy = 0
    best_model_entry = None
    for key in experiments:
        if model_type == '':
            if experiments[key]['accuracy'] > best_model_accuracy:
                best_model_accuracy = experiments[key]['accuracy']
                best_model_entry = experiments[key]
        elif experiments[key]['color_item'] == model_type:
            if experiments[key]['accuracy'] > best_model_accuracy:
                best_model_accuracy = experiments[key]['accuracy']
                best_model_entry = experiments[key]
    return best_model_entry

if __name__ == '__main__':
    experiments_list = main_hog()
    best_models = get_models(experiments_list)
