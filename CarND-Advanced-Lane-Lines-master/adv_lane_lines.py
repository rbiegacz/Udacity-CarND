""" This progam implements Udacity Advanced Lane Lines project """
# imports

from glob import glob
import numpy as np

# importing open cv to process images;
# main functions used: camera calibration, perspective transoform
import cv2

#from keras.layers.convolutional import Convolution2D
#from keras.layers.core import Activation, Dropout
#from keras.layers.normalization import BatchNormalization
#from keras.models import Sequential, model_from_json
#from keras.regularizers import l2

#from moviepy.editor import VideoFileClip

#import tensorflow as tf


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def camera_calibration(image_folder):
    """
    This function calibrates camera
    - image_folder - directory where to find images for calibration
    return: calibration data: mtx, dist

    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

    """
    object_points = np.zeros((6 * 9, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # getting all the images from camera_cal folder
    # all of them will be used to calibrate camera
    cal_files = glob("{}/calibration*.jpg".format(image_folder))

    calibration_images = []
    objpoints = []
    imgpoints = []

    for fname in cal_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        calibration_images.append(gray)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            objpoints.append(object_points)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # uncomment if you would like to see results
            # cv2.imshow('img', img)
            # cv2.waitKey(10000)
        # one needs to distroy all the images
        #  if you decided to see them
        # cv2.destroyAllWindows()
    # ret, mtx, dist, rvecs, tvecs = \
    ret, mtx, dist, _, _ = \
        cv2.calibrateCamera(objpoints, imgpoints,
                            calibration_images[0].shape[::-1], None, None)
    return [mtx, dist]


def distortion_correction(calibration, dir_distorted_images, dir_output_images):
    """ this function remove image distortion """
    distored_files = glob("{}/*.jpg".format(dir_distorted_images))
    for file in distored_files:
        img = cv2.imread(file)
        dst_img = cv2.undistort(img, calibration[0], calibration[1], None, calibration[0])
        file_to_write = dir_output_images+'/'+file.split('\\')[-1]
        cv2.imwrite(file_to_write, dst_img)

    return

def set_color_gradient_thresholds():
    """ setting colors and gradient """
    pass

def perspective_transform():
    """ transoforming image to get proper perspective """
    pass

def detect_lane_lines():
    """ Street lanes detection """
    pass

def determine_lane_curvature():
    """ Determination of lanve curvature """
    pass

def main():
    """
    This function implements the main flow of the application
    Project Steps will be as follows:
    - Camera calibration
    - Distortion correction
    - Color/gradient threshold
    - Perspective transform
    - After doing these steps, you’ll be given two additional steps for the project:
    - Detect lane lines
    - Determine the lane curvature


Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


    """

    # calibrating camera based on images from camera_cal
    calibration = camera_calibration("camera_cal")

    # undistort camera images, store artifacts
    distortion_correction(calibration, "camera_cal", "output_images")

    set_color_gradient_thresholds()

    perspective_transform()

    detect_lane_lines()

    determine_lane_curvature()

    # processing the video
    #clip_in = VideoFileClip(video_path_in)
    #processor = video_processor(lane_model=lane_model,
    # car_model=car_model, calibration=calibration)
    #clip_out = clip_in.fl_image(processor.process_image)
    #clip_out.write_videofile(video_path_out, audio=False)


if __name__ == '__main__':
    main()
