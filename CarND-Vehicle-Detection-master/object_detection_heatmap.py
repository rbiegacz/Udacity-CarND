"""
    Most of the code included in this file comes from Udacity Self-Driving Car Engineer Nanodegree
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    """
    TODO: Deliver description here
    :param heatmap:
    :param bbox_list:
    :return:
    """
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    """
    TODO: Deliver description here
    :param heatmap:
    :param threshold:
    :return:
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    """
    TODO: Deliver description here
    :param img:
    :param labels:
    :return:
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

BOX_LIST = [((800, 400), (900, 500)), ((850, 400), (950, 500)),\
          ((1050, 400), (1150, 500)), ((1100, 400), (1200, 500)),\
          ((1150, 400), (1250, 500)), ((875, 400), (925, 450)),\
          ((1075, 400), (1125, 450)), ((825, 425), (875, 475)),\
          ((814, 400), (889, 475)), ((851, 400), (926, 475)),\
          ((1073, 400), (1148, 475)), ((1147, 437), (1222, 512)),\
          ((1184, 437), (1259, 512)), ((400, 400), (500, 500))]

def main_heatmap(box_list=BOX_LIST):
    """
    TODO: Deliver description here
    :param box_list:
    :return:
    """
    # Read in image similar to one shown above
    image = mpimg.imread('test_images/test1.png')
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()

if __name__ == '__main__':
    main_heatmap()
