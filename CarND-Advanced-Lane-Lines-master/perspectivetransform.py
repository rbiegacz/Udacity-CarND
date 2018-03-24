""" This code implements Udacity Advanced Lane Lines project """
from glob import glob
import numpy as np
import cv2


def perspective_pipeline():
    files_to_transform = glob("output_images\\lines_*.jpg")
    for file in files_to_transform:
        perspective_transform(file, 'output_images\\warped_lines_'+file.split('\\')[-1])
    return


def perspective_transform(src_file, dst_file):
    """
    this function transforms images to birds' view
    :param src_file: file to transform
    :param dst_file: file name of the file under which the results should be stored
    :return: this function doesn't return anything
    """
    img = cv2.imread(src_file)  # Read the test img

    (IMAGE_H, IMAGE_W) = (img.shape[1], img.shape[0])
    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

    img = img[450:(450 + IMAGE_H), 0:IMAGE_W]  # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping
    # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    cv2.imwrite(dst_file, warped_img)
