import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from searchlines import search_for_lines
from perspectivetransform import perspective_transform

def draw_lane(undist, image, warped, left_fitx, right_fitx, ploty, minv):
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
    warped = mpimg.imread('warped_example.jpg')
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center, level_param):
        result = np.zeros_like(img_ref)
        result[int(img_ref.shape[0]-(level_param+1)*height): int(img_ref.shape[0] - level_param * height), max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return result

    def find_window_centroids(image, window_width, window_height, margin):

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
    if len(window_centroids) > 0:

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


def determine_lane_curvature():
    """
    This function calculates curvature for a line in an image
    :return:
    none
    """
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                      for y in ploty])
    rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                       for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = \
        ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5)/np.absolute(2 * left_fit[0])
    right_curverad = \
        ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5)/np.absolute(2 * right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = \
        ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
               right_fit_cr[1]) ** 2) ** 1.5)/np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    output = {}
    output['left_fitx'] = left_fitx
    output['right_fitx'] = right_fitx
    output['ploty'] = ploty
    output['left_curverad'] = left_curverad
    output['right_curverad'] = right_curverad
    return output

def main():
    file_to_process = "output_images/warped_lines_test3.jpg"
    warped = cv2.imread(file_to_process)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    original = cv2.imread("test_images/test3.jpg")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    result, output = search_for_lines(file_to_process)
    yPlotLength = original.shape[0]//3
    _, _, _, minv = perspective_transform("output_images/lines_test3.jpg")
    curvature_output = determine_lane_curvature()
    draw_lane(original, original, warped,
              curvature_output['left_fitx'], curvature_output['right_fitx'],
              curvature_output['ploty'], minv)

    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
    figure.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original image', fontsize=40)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=40)
    ax3.imshow(original)
    ax3.set_title('Original image', fontsize=40)
    ax4.imshow(warped, cmap='gray')
    ax4.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


if __name__ == '__main__':
    main()
