"""
    this module deals with definining a linear SVC model and using it to detect cars
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import object_detection_utils
import object_detection_hog_features


def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, xstart=None, xstop=None):
    """
    TODO: write description of this function
    :param img:
    :param ystart:
    :param ystop:
    :param scale:
    :param svc:
    :param X_scaler:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param spatial_size:
    :param hist_bins:
    :return:
    """
    # img = img.astype(np.float32) / 255
    rectangles = []
    if xstart is None:
        xstart = 0
    if xstop is None:
        xstop=1280

    img_tosearch = img[ystart:ystop, xstart:xstop, :]

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
    else: ctrans_tosearch = img_tosearch

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

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

            if True:
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = object_detection_utils.bin_spatial(subimg, size=spatial_size)
                hist_features = object_detection_utils.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(((xbox_left+xstart, ytop_draw + ystart), (xbox_left+xstart + win_draw, ytop_draw + win_draw + ystart)))
    return rectangles


def main_hog_subsampling(model, img_file="test_images/test1.png"):
    """
    TODO: deliver description of this function
    :return: nothing is return; if function worked correctly it displays an image
    """
    svc = model['linearSVC']
    orient = model["orient"]
    pix_per_cell = model["pix_per_cell"]
    cells_per_block = model["cell_per_block"]
    hist_bins = model["hist_bins"]
    hog_channels = model['hog_channel']
    color_item = model['color_item']
    X_scaler = model['X_scaler']
    spatial_size = (32, 32)

    object_detection_hog_features.print_parameters(color_item, orient, pix_per_cell, cells_per_block, hog_channels)

    test_img = mpimg.imread(img_file)

    rectangles0 = []
    ystart = 400
    ystop = 600
    scale = 1.5
    xstart =  0
    xstop = None
    rectangles0 = find_cars(test_img, ystart, ystop, scale, color_item,
                           hog_channels, svc, X_scaler, orient, pix_per_cell,
                           cells_per_block,spatial_size, hist_bins, xstart, xstop)

    rectangles = []
    ystart = 500
    ystop = 720
    scale = 3
    xstart =  800
    xstop = None
    rectangles = find_cars(test_img[:, :, :], ystart, ystop, scale, color_item,
                           hog_channels, svc, X_scaler, orient, pix_per_cell,
                           cells_per_block,spatial_size, hist_bins, xstart, xstop)
    rectangles1 = []
    ystart = 450
    ystop = 600
    scale = 2
    xstart =  0
    xstop = 200
    rectangles1 = find_cars(test_img, ystart, ystop, scale, color_item,
                            hog_channels, svc, X_scaler, orient, pix_per_cell,
                            cells_per_block,spatial_size, hist_bins,  xstart, xstop)

    rectangles2 = []
    ystart = 450
    ystop = 600
    scale = 1
    xstart =  1000
    xstop = None
    rectangles2 = find_cars(test_img, ystart, ystop, scale, color_item,
                            hog_channels, svc, X_scaler, orient, pix_per_cell,
                            cells_per_block,spatial_size, hist_bins,  xstart, xstop)

    rectangles3 = []
    ystart = 400
    ystop = 650
    scale = 2
    xstart =  100
    xstop = 900
    rectangles3 = find_cars(test_img, ystart, ystop, scale, color_item,
                            hog_channels, svc, X_scaler, orient, pix_per_cell,
                            cells_per_block,spatial_size, hist_bins,  xstart, xstop)

    display_rectangles = rectangles+rectangles0+rectangles1+rectangles2+rectangles3
    #display_rectangles = rectangles0 + rectangles
    print("Number of rectangles: {}".format(len(rectangles0)))
    print("Number of rectangles: {}".format(len(rectangles)))
    print("Number of rectangles: {}".format(len(rectangles1)))
    print("Number of rectangles: {}".format(len(rectangles2)))
    print("Number of rectangles: {}".format(len(rectangles3)))
    pic = object_detection_utils.draw_boxes(cv2.imread(img_file), display_rectangles)
    # you can replace cv2.imread(...) with mpimg.imread(img_file)
    # but you will need to comment out the line below
    pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    plt.imshow(pic)

