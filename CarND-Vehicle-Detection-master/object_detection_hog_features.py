"""
This module contains functions for HOG feature extraction
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


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    TODO: description what this function does
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    # Call with two outputs if vis==True
    if vis is True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
    return features


def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """
    This function extracts features from a list of images
    Have this function call bin_spatial() and color_hist()
    :param imgs:
    :param cspace:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


def print_parameters(colourspace, orient, pix_per_cell, cell_per_block, hog_channel):
    print("Color Space: {:6.6}, Hog Channel: {}, Orient: {}, Pix per Cell: {:2}, Cell per Block: {}: calculation".format(colourspace, hog_channel,  orient, pix_per_cell, cell_per_block))


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
    sample_size = 500
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    experiments_dict = dict()
    color_space = {'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'}
    orients = [8, 9, 10, 11] #[7, 8, 9, 10, 11]
    pixels_per_block = [8, 16] #[4, 8, 12, 16]
    cells_per_block = [2] #[2, 4, 8]
    hog_channels = [0] #[0, 1, 2, "ALL"]
    for color_item in color_space:
        for orient in orients:
            for pix_per_cell in pixels_per_block:
                for cell_per_block in cells_per_block:
                    for hog_channel in hog_channels:
                        print_parameters(color_item, orient, pix_per_cell, cell_per_block, hog_channel)
                        t = time.time()
                        car_features = extract_features(cars, cspace=color_item, orient=orient,
                                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                        hog_channel=hog_channel)
                        notcar_features = extract_features(not_cars, cspace=color_item, orient=orient,
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
                        t = time.time()
                        n_predict = 10
                        t2 = time.time()
                        time_for_prediction = round(t2 - t, 5)
                        experiments_dict["{}_{}_{}_{}_{}".format(color_item, hog_channel, orient, pix_per_cell, cell_per_block)] = \
                            {"color_item": color_item, "hog_channel": hog_channel,
                             "orient": orient, "pix_per_cell":pix_per_cell,
                             "cell_per_block": cell_per_block,
                             "time_for_prediction": time_for_prediction,
                             "time_extract_hog_features": time_extract_hog_features,
                             "accuracy": accuracy,
                             "feature vector length": len(X_train[0]),
                             "linearSVC": svc}
    return experiments_dict


def print_results(experiments_dict):
    print("CS means Color Space")
    print("Or means Orientation")
    print("C/B means Cells per Block")
    print("P/C means Pixels per Cell")
    print("\n")
    # T(pred):{}\t| T(ext):{}\t|
    for key in experiments_dict:
        results = experiments_dict[key]
        print("| CS:{}\t| Or:{:02}\t| C/B:{}\t| P/C:{:02}\t| Acc {:06.4f}\t| F:{:06}\t|"\
              .format(results["color_item"], results["orient"],
                      results["cell_per_block"], results["pix_per_cell"], results["accuracy"],
                      # results["time_for_prediction"], results["time_extract_hog_features"],
                      results["feature vector length"]))


if __name__ == '__main__':
    experiments_dict = main_hog()
