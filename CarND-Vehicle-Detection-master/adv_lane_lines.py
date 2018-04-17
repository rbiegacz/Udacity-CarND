""" This progam implements Udacity Advanced Lane Lines project """

from glob import glob
from perspectivetransform import perspective_pipeline
from searchlines import detect_lane_lines
from searchlines import thresholds_pipeline
from correctcamera import camera_calibration
from correctcamera import distortion_correction
import roadlanes

def main():
    """
    This function implements the main flow of the application
    Project Steps will be as follows:
    - Camera calibration
    - Distortion correction
      Apply a distortion correction to raw images.
    - Color/gradient threshold
      Use color transforms, gradients, etc., to create a thresholded binary image.
    - Perspective transform
      Apply a perspective transform to rectify binary image ("birds-eye view").
    - After doing these steps, you’ll be given two additional steps for the project:
    - Detect lane lines
      Detect lane pixels and fit to find the lane boundary.
    - Determine the lane curvature
      Determine the curvature of the lane and vehicle position with respect to center.
    - Warp the detected lane boundaries back onto the original image.
      Output visual display of the lane boundaries and numerical
      estimation of lane curvature and vehicle position.
    """

    # calibrating camera based on images from camera_cal
    calibration = camera_calibration("camera_cal")
    # files_to_correct = glob("camera_cal/*.jpg")
    # distortion_correction(calibration, files_to_correct, "output_images")

    # remove distortion from test images and store the resulting images in output_images folder
    files_to_correct = glob("test_images/*.jpg")
    distortion_correction(calibration, files_to_correct, "output_images")

    # discovering lines
    # applying Sobel gradient and using HLS color map
    # stacking two methods over each other
    # storing the results of this transformation in "output_images" folder
    files_to_transform = glob("output_images/undist*.jpg")
    thresholds_pipeline(files_to_transform, "output_images")

    # change of perspective to birds' view
    perspective_pipeline()

    # detect lines in the warped images
    detect_lane_lines()

    # draw a lane between lane lines
    roadlanes.draw_lane_pipeline()

    # processing the video
    # uncomment the line below if you want to run video processing
    roadlanes.main_video()

if __name__ == '__main__':
    main()
