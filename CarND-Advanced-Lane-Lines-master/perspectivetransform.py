""" This code implements Udacity Advanced Lane Lines project """
from glob import glob
import numpy as np
import cv2
from utils import display_two_images

def perspective_pipeline():
    """
    this function iterates thru all the images and applies birds' view transformation
    :return: it doesn't return anything if everything is fine
    """
    files_to_transform = glob("output_images\\lines_*.jpg")
    for file in files_to_transform:
        warped, unwarped, _, _ = \
            perspective_transform(file)
        # warped = warped[2*warped.shape[0]//3:warped.shape[0],100:warped.shape[0]-100,:]
        # warped = cv2.resize(warped, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite('output_images\\warped_'+file.split('\\')[-1], warped)
        cv2.imwrite('output_images\\unwarped_'+file.split('\\')[-1], unwarped)
    return


def perspective_transform(src_file, image=None):
    """
    this function transforms images to birds' view
    :param src_file: file to transform
    :return: warped, unwarped images, matrices: M and MInv
    """

    if src_file:
        img = cv2.imread(src_file)  # Read the test img
    else:
        img = image

    src = np.float32([[545, 460], [735, 460], [1280, 700], [0, 700]])
    dst = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])

    m = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation
    warped_img = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    unwarped_img = \
        cv2.warpPerspective(warped_img, minv, (warped_img.shape[1], warped_img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped_img, unwarped_img, m, minv


def main():
    """
    main() shows how to use functions to achieve bird's eye perspective
    :return:
    """
    files_to_transform = "output_images/lines_undist_straight_lines1.jpg"
    warped, _, _, _ = perspective_transform(files_to_transform)
    #warped = warped[2 * warped.shape[0] // 3:warped.shape[0], 100:warped.shape[0] - 100, :]
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    original_img = cv2.imread(files_to_transform)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    display_two_images(original_img, "Binary with Lines", warped, "Bird's eye perspective")

if __name__ == '__main__':
    main()
