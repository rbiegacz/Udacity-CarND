""" This code implements Udacity Advanced Lane Lines project """
from glob import glob
import numpy as np
import cv2


def perspective_pipeline():
    """
    this function iterates thru all the images and applies birds' view transformation
    :return: it doesn't return anything if everything is fine
    """
    files_to_transform = glob("output_images\\lines_*.jpg")
    for file in files_to_transform:
        warped, unwarped, _, _ = \
            perspective_transform(file)
        warped = warped[2*warped.shape[0]//3:warped.shape[0],100:warped.shape[0]-100,:]
        # warped = cv2.resize(warped, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite('output_images\\warped_'+file.split('\\')[-1], warped)
        cv2.imwrite('output_images\\unwarped_'+file.split('\\')[-1], unwarped)
    return


def perspective_transform(src_file):
    """
    this function transforms images to birds' view
    :param src_file: file to transform
    :return: warped, unwarped images, matrices: M and MInv
    """

    img = cv2.imread(src_file)  # Read the test img

    img_height = img.shape[0]
    img_width = img.shape[1]

    perspective_delta_x = 744
    perspective_delta_y = int(perspective_delta_x * 30 / 3.7)
    perspective_border_x = int(perspective_delta_x * 0.7)
    perspective_max_y = perspective_delta_y
    perspective_max_x = int(perspective_delta_x + 2 * perspective_border_x)
    perspective_pixels_per_meter = perspective_delta_x / 3.7
    # print("X: {}".format(perspective_max_x))
    # print("Y: {}".format(perspective_max_y))

    perspective_origin_y_top = 440
    perspective_origin_y_bottom = 670
    perspective_origin_x_top_left = 609
    perspective_origin_x_top_right = 673
    perspective_origin_x_bottom_left = 289
    perspective_origin_x_bottom_right = 1032

    src = np.float32(
        [[perspective_origin_x_top_left, perspective_origin_y_top],
         [perspective_origin_x_top_right, perspective_origin_y_top],
         [perspective_origin_x_bottom_left, perspective_origin_y_bottom],
         [perspective_origin_x_bottom_right, perspective_origin_y_bottom]])

    dst = np.float32(
        [[perspective_border_x, 0],
         [perspective_border_x + perspective_delta_x, 0],
         [perspective_border_x, perspective_delta_y],
         [perspective_border_x + perspective_delta_x, perspective_delta_y]])

    m = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation
    warped_img = cv2.warpPerspective(img, m, (perspective_max_x, perspective_max_y), flags=cv2.INTER_LINEAR)
    unwarped_img = \
        cv2.warpPerspective(warped_img, minv, (img.shape[0], img.shape[1]), flags=cv2.INTER_LINEAR)
    return warped_img, unwarped_img, m, minv
