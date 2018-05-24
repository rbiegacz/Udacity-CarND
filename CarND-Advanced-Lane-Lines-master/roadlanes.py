""" this module draws a road lane in a single image and in video """

import cv2
from glob import glob
import numpy as np
import imageio
imageio.plugins.ffmpeg.download()
import matplotlib.pyplot as plt
from pathlib import Path

from utils import display_two_images
from correctcamera import camera_calibration, distortion_removal
from searchlines import search_for_lines, apply_gradients_thresholds
from perspectivetransform import perspective_transform
from moviepy.editor import VideoFileClip


# from Line import Line
# from line_fit import line_fit, tune_fit, calc_curve, calc_vehicle_offset


def draw_lane(undist, image, warped, left_fitx, right_fitx, ploty, minv):
    """
    TODO: deliver description
    :param undist:
    :param image:
    :param warped:
    :param left_fitx:
    :param right_fitx:
    :param ploty:
    :param minv:
    :return:
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def convolutions():
    """
    calculation of lines
    :return:
    """
    # Read in a thresholded image
    warped = cv2.imread('output_images/warped_lines_undist_straight_lines1.jpg')
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center, level_param):
        """ TODO: deliver description """
        result = np.zeros_like(img_ref)
        result[int(img_ref.shape[0]-(level_param+1)*height): int(img_ref.shape[0] - level_param * height), max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return result

    def find_window_centroids(image, window_width, window_height, margin):
        """ TODO: deliver description """
        # Store the (left,right) window centroid positions per level
        window_centroids = []
        # Create our window template that we will use for convolutions
        window = np.ones(window_width)

        # First find the two starting positions for the left
        # and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - \
                   window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[int(image.shape[0] - (level + 1) * window_height):
                      int(image.shape[0] - level * window_height), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal
            # reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if window_centroids:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height,
                                 warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height,
                                 warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | (l_mask == 1)] = 255
            r_points[(r_points == 255) | (r_mask == 1)] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((warped, warped, warped)) * 255  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()


def determine_lane_curvature(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    """
    TODO: deliver description
    :param left_lane_inds:
    :param right_lane_inds:
    :param nonzerox:
    :param nonzeroy:
    :return:
    """
    y_eval = 719  # 720p video/image, so last (lowest on screen) y index is 719

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate radius of curvatures
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    output = {}
    output['left_curverad'] = left_curverad
    output['right_curverad'] = right_curverad
    return output

def vehicle_position(image, output):
    # Calculate vehicle center
    xm_per_pix = 3.7 / 700
    ym_per_pix = 30 / 720
    xMax = image.shape[1]
    yMax = image.shape[0]
    vehicleCenter = xMax / 2
    left_fit_m = output['left_lane_inds']
    right_fit_m = output['right_lane_inds']
    lineLeft = left_fit_m[0]
    lineRight = right_fit_m[0]
    car_position = xm_per_pix*1280/2 - (lineRight + lineLeft) / 2
    if car_position >= 0:
        vehicleposition = 'Vehicle position from lane center: {:.2f} m right'.format(car_position)
    else:
        vehicleposition = 'Vehicle position from lanen center: {:.2f} m left'.format(-car_position)
    return vehicleposition

def annotate_movie(input_video=None, output_video=None):
    """
    this funcion annotates a movice - takes one frame at a time and annotates it
    :param input_video: name of the video to annotate (mp4)
    :param output_video: name of the video to store the annotated video (mp4)
    :return:
    """
    # get data for calibration
    calibration = camera_calibration("camera_cal")

    def annotate_image(image, image_file=None):
        """
        this function is used to annotate each video frame
        this function can annotate either an image or it can read image from a file
        :param image: image to annotate
        :param image_file: file name of the image to annotate
        :return: annotated image/frame
        """
        # if not image and not image_file:
        #    raise ValueError("annotate_image: wrong function arguments (both of them are null")
        if image.any() and image_file:
            raise ValueError("annotate_image: wrong function arguments (both of them are not null)")
        if not image.any():
            raise NotImplementedError("this function accepts only input in the form of image")

        # removing distortion
        undistorted = distortion_removal(calibration, imageFile=None, image=image)

        # discovering lines
        gradient = apply_gradients_thresholds(image=undistorted)

        # changing perspective
        warped, _, _, minv = perspective_transform(src_file=None, image=gradient)
        _, output = search_for_lines(img=warped, file_image=None)

        # discovering curvature
        curvature_output = \
            determine_lane_curvature(output['left_lane_inds'],
                                     output['right_lane_inds'],
                                     output['nonzerox'],
                                     output['nonzeroy'])

        car_position_msg = vehicle_position(image, output)

        # drawing lane & annotating the image
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        res = draw_lane(image, undistorted, warped, output['left_fitx'], output['right_fitx'], output['ploty'], minv)
        avg_curve = (curvature_output['left_curverad'] + curvature_output['right_curverad']) / 2
        label_curve = 'Radius of curvature: %.1f m' % avg_curve
        res = cv2.putText(res, label_curve, (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        res = cv2.putText(res, car_position_msg, (30, 80), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return res

    if input_video:
        video = VideoFileClip(input_video)
        annotated_video = video.fl_image(annotate_image)
        annotated_video.write_videofile(output_video, audio=False)
    else:
        orig_image_name = "test_images/test3.jpg"
        original = cv2.imread(orig_image_name)
        result = annotate_image(image=original)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result)
        plt.show()
    return


def main_video():
    annotate_movie("project_video.mp4", "annotated_project_video.mp4")
    # annotate_movie("challenge_video.mp4", "annotated_challenge_video.mp4")


def draw_lane_pipeline(files=None, display_images=False):
    """ this is the function that processes single image pointed by a name mentioned below """
    # orig_image_name = "straight_lines1.jpg"
    if files is None:
        files_to_process = glob("test_images/*.jpg")
    else:
        files_to_process = list()
        files_to_process.append(files)

    for file in files_to_process:
        if "\\" in file:
            file_distortion_corrected = "output_images/lines_undist_{}".format(file.split('\\')[-1])
            file_to_process = "output_images/warped_lines_undist_{}".format(file.split('\\')[-1])
        else:
            file_distortion_corrected = "output_images/lines_undist_{}".format(file.split('/')[-1])
            file_to_process = "output_images/warped_lines_undist_{}".format(file.split('/')[-1])

        original = cv2.imread(file)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        undistorted = cv2.imread(file_distortion_corrected)
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

        warped, _, _, minv = perspective_transform(file_distortion_corrected)
        result, output = search_for_lines(file_to_process)

        # Calculate vehicle center
        vehicleposition_msg = vehicle_position(original, output)

        curvature_output = \
            determine_lane_curvature(output['left_lane_inds'],
                                     output['right_lane_inds'],
                                     output['nonzerox'],
                                     output['nonzeroy'])
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        result = draw_lane(original, undistorted, warped, output['left_fitx'], output['right_fitx'], output['ploty'], minv)
        avg_curve = (curvature_output['left_curverad'] + curvature_output['right_curverad']) / 2
        label_curve = 'Radius of curvature: %.1f m' % avg_curve
        result = cv2.putText(result, label_curve, (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        result = cv2.putText(result, vehicleposition_msg, (30, 80), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        file_to_write = 'output_images/annotated_'+file.split('\\')[-1]
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if display_images:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.imshow(result)
            plt.show()
        else:
            cv2.imwrite(file_to_write, result)


def main_image():
    """ this is the function that processes single image pointed by a name mentioned below """
    file_exists = Path("test_images\\test3.jpg")
    if file_exists.is_file():
      draw_lane_pipeline("test_images\\test3.jpg", display_images=True)
    else:
      draw_lane_pipeline("test_images/test3.jpg", display_images=True)

if __name__ == '__main__':
    main_video()
