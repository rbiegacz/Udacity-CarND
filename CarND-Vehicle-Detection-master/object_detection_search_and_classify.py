"""
    This module implements a mechanism of searching and clasification of objects using sliding windows;
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""
import glob
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import object_detection_utils


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    """
    TODO: write a description of this function
    :param img:
    :param size:
    :return:
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    TODO: write a description of this function
    # Define a function to compute color histogram features
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    :param img:
    :param nbins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    TODO: write a description for this function
    Define a function to extract features from a single image window
    This function is very similar to extract_features() just for a single image rather than list of images
    :param img:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
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
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                features = object_detection_utils.get_hog_features(feature_image[:, :, channel],
                                                                   orient, pix_per_cell, cell_per_block,
                                                                   vis=False, feature_vec=True)
                hog_features.extend(features)
        else:
            hog_features = \
                object_detection_utils.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                        pix_per_cell, cell_per_block,
                                                        vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """
    TODO: write a description of this function
    Define a function you will pass an image
    and the list of windows to be searched (output of slide_windows())
    :param img:
    :param windows:
    :param clf:
    :param scaler:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param hist_range:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def main_search_and_classify():
    """
    TODO: write a description of this function
    :return:
    """
    # Read in cars and notcars
    # images = glob.glob('test_images//*.jpeg')
    cars = glob.glob('util_images/vehicles/**/*.png')
    notcars = glob.glob('util_images/non-vehicles/**/*.png')

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 1000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [400, 750]  # Min and max in y to search in slide_window()

    car_features = object_detection_utils.extract_features(cars, color_space=color_space,\
                                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                                           orient=orient, pix_per_cell=pix_per_cell,
                                                           cell_per_block=cell_per_block,
                                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                           hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = \
        object_detection_utils.extract_features(notcars, color_space=color_space,
                                                spatial_size=spatial_size, hist_bins=hist_bins,
                                                orient=orient, pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    image = mpimg.imread('util_images/bbox-example-image.png')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    windows = object_detection_utils.slide_window(image, x_start_stop=[None, None],
                                                  y_start_stop=y_start_stop,
                                                  xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = object_detection_utils.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)

if __name__ == '__main__':
    main_search_and_classify()
