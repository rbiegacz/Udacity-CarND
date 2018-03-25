""" you can find here function for camera correction """

from glob import glob
import numpy as np
import cv2


def camera_calibration(image_folder):
    """
    This function calibrates camera. It computes the camera calibration matrix and distortion coefficients
    given a set of chessboard images.
    :param image_folder: directory where to find images for calibration
    :return: calibration data: mtx, dist
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
            # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
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
    distorted_files = glob("{}/*.jpg".format(dir_distorted_images))
    distorted_files += glob("test_images/*.jpg")
    for file in distorted_files:
        img = cv2.imread(file)
        dst_img = cv2.undistort(img, calibration[0], calibration[1], None, calibration[0])
        file_to_write = dir_output_images+'/undist_'+file.split('\\')[-1]
        cv2.imwrite(file_to_write, dst_img)
    return
