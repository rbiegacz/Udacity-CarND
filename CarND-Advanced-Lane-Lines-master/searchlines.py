""" here you can find functions for line detection """

from glob import glob
import numpy as np
import cv2
from utils import display_two_images

class Line:
    """
    this class stores information about a line
    """
    Counter = 0
    N_Average = 20
    def __init__(self):
       # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
       # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def get_current_fit(self):
        return self.current_fit

    def get_best_fit(self):
        return self.best_fit

    def add_fit(self, fit_coeffs):
        # add current coofiecients to the list
        # x values of the last n fits of the line
        self.recent_xfitted[Line.Counter%Line.N_Average] = fit_coeffs
        # difference in fit coefficients between last and new fits
        self.diffs = self.best_fit - self.current_fit
        # polynomial coefficients averaged over the last n iterations
        self.best_fit= np.average(self.recent_xfitted, 0)
        # average x values of the fitted line over the last n iterations
        self.bestx = self.best_fit[0]*self.ally**2 + self.best_fit[1]*self.ally + self.best_fit[2]
        Line.Counter = (Line.Counter+1)%Line.N_Average

def apply_gradients_thresholds(image_file=None, s_thresh=(170, 255), sx_thresh=(20, 100), image=None):
    """
    this function generates image with lines by application of Sobel operator and using HSL color map
    :param image_file:
    :param s_thresh:
    :param sx_thresh:
    :param image:
    :return:
    """
    if image_file:
        img = cv2.imread(image_file)
    else:
        img = image
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = hls.astype(np.float)
    s_channel = hls[:, :, 2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x for a grayed image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    gray_binary = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
    return gray_binary


def thresholds_pipeline(source_files, target_dir, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    This function uses gradients and color maps to extract line features
    :param source_dir: folder with images to process
    :param target_dir: folder the results will be stored
    :param s_thresh: thresholds for S channel (HLS color scheme)
    :param sx_thresh: Sobel X transformation
    :return: processed image
    """
    for file in source_files:
        gray_binary = apply_gradients_thresholds(file, s_thresh, sx_thresh)
        file_to_write = target_dir+'/lines_'+file.split('\\')[-1]
        cv2.imwrite(file_to_write, gray_binary)
    return


def detect_lane_lines():
    """
    A loop that processes all the files found in output_images directory.
    Street lanes detection is done on each found image.
    :return: the function doesn't return anything if everything is OK
    """
    files_to_transform = glob("output_images/warped_lines_*.jpg")
    for file in files_to_transform:
        result, _ = search_for_lines(file)
        file_to_write = 'output_images/per_'+file.split('\\')[-1]
        cv2.imwrite(file_to_write, result)


def search_for_lines(file_image, img=None):
    """
    This functions searches for lines in a given picture.
    :return:
    processes image
    """
    # Assuming you have created a warped binary image called "binary_warped"
    if file_image:
        binary_warped = cv2.imread(file_image)
        binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    else:
        binary_warped = img
        binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = int(leftx_current - margin)
        win_xleft_high = int(leftx_current + margin)
        win_xright_low = int(rightx_current - margin)
        win_xright_high = int(rightx_current + margin)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds =\
            ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    how_long_lane = 2
    ystart = 0
    ploty = np.linspace(ystart, (binary_warped.shape[0]-1-ystart)//how_long_lane, (binary_warped.shape[0]-ystart)//how_long_lane)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = \
        ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                      left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                            left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = \
        ((nonzerox > (right_fit[0]*(nonzeroy**2) +
                      right_fit[1]*nonzeroy +
                      right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                             right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    how_long_lane = 1
    ystart = 0
    ploty = np.linspace(ystart,
                        (binary_warped.shape[0]-1-ystart)//how_long_lane,
                        (binary_warped.shape[0]-ystart)//how_long_lane)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    output = dict()
    output['left_fit'] = left_fit
    output['right_fit'] = right_fit
    output['left_fitx'] = left_fitx
    output['right_fitx'] = right_fitx
    output['ploty'] = ploty
    output['nonzerox'] = nonzerox
    output['nonzeroy'] = nonzeroy
    output['left_lane_inds'] = left_lane_inds
    output['right_lane_inds'] = right_lane_inds
    return None, output


def main():
    """
    this is main function that shows how to use functions from this python module/file
    :return:
    """
    files_to_transform = "output_images/undist_straight_lines1.jpg"
    transformed_image = apply_gradients_thresholds(files_to_transform)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

    original_img = cv2.imread(files_to_transform)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    display_two_images(original_img, 'Corrected image', transformed_image, 'Binary image')

if __name__ == '__main__':
    main()
