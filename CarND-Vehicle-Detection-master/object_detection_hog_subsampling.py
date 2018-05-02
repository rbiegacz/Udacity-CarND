"""
    this module deals with definining a linear SVC model and using it to detect cars
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""
import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import object_detection_utils
import object_detection_hog_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog



def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # select colorspace channel for HOG
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else:
        ch1 = ctrans_tosearch[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = object_detection_utils.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = object_detection_utils.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = object_detection_utils.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            # subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            # hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            # print(spatial_features.shape)
            # print(hist_features.shape)
            # print(hog_features.shape)
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features)).reshape(1, -1))
            #test_prediction = svc.predict(test_features)
            hog_features = hog_features.reshape(-1, 1)
            test_prediction = svc.predict(hog_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
    return draw_img


def main_hog_subsampling(model, img_file="test_images/test1.jpg"):
    """
    TODO: deliver description of this function
    :return: nothing is return; if function worked correctly it displays an image
    """
    #svc = model['linearSVC']
    cars = glob.glob('util_images/vehicles/**/*.png')
    not_cars = glob.glob('util_images/non-vehicles/**/*.png')
    sample_size = 100
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    orient = model["orient"]
    pix_per_cell = model["pix_per_cell"]
    cells_per_block = model["cell_per_block"]
    spatial_size = model["spatial_size"]
    hist_bins = model["hist_bins"]
    hog_channels = model['hog_channel']
    color_item = model['color_item']
    test_img = mpimg.imread(img_file)
    ystart = 400
    ystop = 656
    scale = 1.5

    object_detection_hog_features.print_parameters(color_item, orient, pix_per_cell, cells_per_block, hog_channels)

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

    out_img = find_cars(test_img, ystart, ystop, scale, color_item, hog_channels,
                        svc, X_scaler, orient, pix_per_cell, cells_per_block,
                        spatial_size, hist_bins)

    #print(len(rectangles), 'rectangles found in image')
    plt.imshow(out_img)

