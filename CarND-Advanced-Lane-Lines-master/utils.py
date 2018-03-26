""" this module contains util functions used in other project python files """

import matplotlib.pyplot as plt


def display_two_images(img1, title1, img2, title2):
    """
    this function draws two images side by side
    :param img1: the first image to draw
    :param title1: title of the first image
    :param img2: the second image to draw
    :param title2: title of the second image
    :return: this function doesn't return anything
    """
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    figure.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=40)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    plt.close()
