""" This progam implements Udacity Advanced Lane Lines project """

from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from perspectivetransform import perspective_pipeline
from searchlines import detect_lane_lines
from searchlines import thresholds_pipeline
from correctcamera import camera_calibration
from correctcamera import distortion_correction

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

    # remove distortion from camera images, store artifacts
    distortion_correction(calibration, "camera_cal", "output_images")

    # discovering lines
    # applying Sobel gradient and using HLS color map
    # stacking two methods over each other
    # storing the results of this transformation in "output_images" folder
    thresholds_pipeline("test_images", "output_images")

    # change of perspective to birds' view
    perspective_pipeline()

    # detect lines in the warped images
    detect_lane_lines()

    # draw a lane between lane lines
    # draw_lane()

    # calculate curvatures of lines
    # determine_lane_curvature()

    # processing the video
    # clip_in = VideoFileClip(video_path_in)
    # processor = video_processor(lane_model=lane_model,
    # car_model=car_model, calibration=calibration)
    # clip_out = clip_in.fl_image(processor.process_image)
    # clip_out.write_videofile(video_path_out, audio=False)


if __name__ == '__main__':
    main()
