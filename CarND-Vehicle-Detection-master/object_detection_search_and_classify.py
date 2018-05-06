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
import object_detection_heatmap
import object_detection_hog_subsampling


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
        #feature_image = np.copy(img)
        feature_image = img
    # 3) Compute spatial features if flag is set
    if spatial_feat is True:
        spatial_features = object_detection_utils.bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat is True:
        hist_features = object_detection_utils.color_hist_features(feature_image, nbins=hist_bins)
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
    #print(windows)
    for window in windows:
        # 3) Extract the test window from original image
        test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        test_img = cv2.resize(test_img, (64, 64))
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


def main_search_and_classify_many(model=None, image_files=None, use_heatmap=False,
                                  display_results=False, method="both"):
    images = []
    k = 1
    fig, axs = plt.subplots(3, 2, figsize=(20, 40))
    axs = axs.ravel()
    for image_file in image_files:
        img = main_search_and_classify(model, None, image_file=image_file,
                                       use_heatmap=use_heatmap, display_results=False,
                                       method=method)

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if use_heatmap:
            cv2.imwrite("output_images/heatmap_processed_{}".format(image_file.split('/')[-1]), 255*img_RGB)
        else:
            cv2.imwrite("output_images/processed_{}".format(image_file.split('/')[-1]), 255*img_RGB)
        plt.title('Image {}'.format(k))
        axs[k-1].axis('off')
        axs[k-1].set_title('Image {}'.format(k), fontsize=10)
        axs[k-1].imshow(img)
        k += 1


def main_search_and_classify(model=None, image=None, image_file='util_images/bbox-example-image.png',
                             use_heatmap=False, display_results=True, method="both"):
    """
    TODO: write a description of this function
    :return:
    """
    # Read in cars and notcars
    # images = glob.glob('test_images//*.jpeg')
    svc = model["linearSVC"]
    X_scaler = model["X_scaler"]
    color_space = model["color_item"]
    orient = model["orient"]
    pix_per_cell = model["pix_per_cell"]
    cell_per_block = model["cells_per_block"]
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 2*pix_per_cell  # Number of histogram bins
    hog_channel = model["hog_channel"]
    spatial_feat = model["spatial_feat"]  # Spatial features on or off
    hist_feat = model["hist_feat"]  # Histogram features on or off
    hog_feat = model["hog_feat"]  # HOG features on or off

    if image_file is not None:
        image = mpimg.imread(image_file)
    # draw_image = np.copy(image)
    draw_image = image
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    hotty_windows = []

    if method in ["both", "sliding_windows"]:
        windows_list = []
        x_start_stop_list = [[0, 400], [410, 850], [860, None]]
        y_start_stop_list = [(450, 70), (420, 500), (450, 700)]
        xy_window_list = [(128, 128), (64, 64), (128, 128)]

        x_start_stop_list = [[50, 1280], [50, 1280], [860, 1200]]
        y_start_stop_list = [(410, 450), (400, 600), (500, 700)]
        xy_window_list = [(32, 32), (64, 64), (128, 128)]
        overlapx = 0.75
        overlapy = 0.75
        overlap_list = [(overlapx, overlapy), (overlapx, overlapy), (overlapx, overlapy)]

        for x, y, xy, xy_overlap in zip(x_start_stop_list, y_start_stop_list, xy_window_list, overlap_list):
            windows_list += (object_detection_utils.slide_window(image, x_start_stop=x,
                                                                 y_start_stop=y, xy_window=xy,
                                                                 xy_overlap=xy_overlap))
        hot_windows = search_windows(image, windows_list, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)
        hotty_windows = hot_windows

    if method in ["both", "hot_windows"]:
        xstart_list = [200, 200, 1000, 400, 400]
        xstop_list = [1280, 900, 1280, 800, 1280]
        ystart_list = [420, 420, 450, 450, 500]
        ystop_list = [600, 650, 600, 600, 720]
        scale_list = [1.5, 2.0, 1, 2, 3]
        rectangles = []
        for ystart, ystop, scale, xstart, xstop in zip(ystart_list, ystop_list, scale_list, xstart_list, xstop_list):
            rectangles += \
                object_detection_hog_subsampling.find_cars(image, ystart, ystop, scale, color_space,
                                                           hog_channel, svc, X_scaler, orient, pix_per_cell,
                                                           cell_per_block,spatial_size, hist_bins, xstart, xstop)
        hotty_windows += rectangles

    if use_heatmap:
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = object_detection_heatmap.add_heat(heat, hotty_windows)
        # Apply threshold to help remove false positives
        heat = object_detection_heatmap.apply_threshold(heat, 4)
        # Visualize the heatmap when displaying
        #heatmap = np.clip(heat, 0, 255)
        heatmap = np.clip(heat, 0, 1)
        # Find final boxes from heatmap using label function
        labels = object_detection_heatmap.label(heatmap)
        #draw_img = object_detection_heatmap.draw_labeled_bboxes(np.copy(image), labels)
        draw_img = object_detection_heatmap.draw_labeled_bboxes(image, labels)
        if display_results:
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
    else:
        draw_img = object_detection_utils.draw_boxes(image, hotty_windows, color=(0, 0, 255), thick=6)
        if display_results:
            fig = plt.figure()
            plt.subplot(211)
            plt.imshow(draw_img)
            plt.title('Identified objects')
    return draw_img

if __name__ == '__main__':
    main_search_and_classify()
