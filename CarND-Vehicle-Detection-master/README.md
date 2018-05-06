# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The main goal in this project is to write a software pipeline to detect vehicles present in videos recorded by car camera (test_video.mp4 and project_video.mp4).

The Project
---

There are following phases implemented within this project:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train Linear SVM classifier to identify vehicles in the images.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Estimate a bounding box around detected vehicles.
* Run processing pipeline on a video stream to detect vehicles in video frames.


Solution Overview
---
This README is accompanied with Jupyter Notebook and a set of Python files that are realization of this project.

Name of Jupyter Notebook is vehicle_detection.ipynb.

The main function responsible for processing of video frames is called main_search_and_classify(...) and is implemented in object_detection_search_and_classify.py

Python files used in this project:

- object_detection_car_notcar.py
This file helps to explore image dataset used for training.

- object_detection_color_classify.py
- object_detection_colorspaces.py
Both these files help to explore how using various color spaces could be helpful in vehicle detection.

- object_detection_fecvector.py

- object_detection_findmatches.py
This file shows how static template matching (cv2.matchTemplate) could be helpful in vehicle identification.

- object_detection_heatmap.py
This file shows how heatmap mechanism might be helpful in object identification.

- object_detection_hog_features.py
- object_detection_hog_subsampling.py
Both these files use HOG features to train classifiers and the use these models to detect vehicles.
main_hog(...) does training of models with different parameters:
- different color spaces: RGB, HSV, HSL, YUV, YCrCb
- number of pixels per cell [values explored: 8, 16]
- number of orients [values explored: 8, 9, 10, 11]

All HOG channels are used always.
Constant number of cells per block is used and is set to 2.

- object_detection_search_and_classify.py
This Python file is used to implement a method used to identify vehicles in the images.

- object_detection_sliding_window.py
This file helps to explore strategies for dividing an image into sub-images. This approach helps to skip these parts of the image that are irrelevant to vehicle detection.
sliding_window_main() shows one of the strategies to how to split an image into parts that could be further processed.

- object_detection_utils.py
This file contains functions shared/used by above-mentioned Python files (e.g. draw_boxes() function)

Summary of the used methods
---
### Data set and classification/identification method
Data set containing vehicles and non-vehicles consists of
- 8792  cars and
- 8968  non-cars

Dataset exploration is implemented in object_detection_car_notcar.py file and is shown in vehicle_detection.ipynb in cells #6-#9

All the images are used in .png format to avoid problems related to different interpretation of .jpg and .png data.

Linear Support Vector Classification (implemented within Scikit Learn library) is used to train classifier to detect vehicles. The whole training set is used to train the classifier.

Feature vector used as an input to Linear SVC consists of the following elements:
- HOG feature vector
- Histogram of image colors 
- Picture pixels

###Histogram of Oriented Gradients (HOG)
This projected uses Histogram of Oriented Gradients (HOG) to detect objects.

HOG feature vector contains:
- computing the gradient image in x and y
- computing gradient histograms
- normalising across blocks
HOG feature vector is flattened into a one-dimensional feature vector.

###Sliding Window Search
The pictures taken by camera were divided into small pieces as it was shown in cell #

###Heatmap
Heatmap mechanism was used to identify areas detected as 'vehicles' by many boxes. This method allowed to eliminate a lot of false positives.

###Video Processing
Video processing is implemented in the last cells of the Jupyter Notebook in the section called "Video Processing".

Processing is done frame by frame and the output is stored in the form of mp4 file. 

Main function responsible for processing of each video frame is called

Summary and Improvement Suggestions
---
###Project Conclusions
Project is a good exploration exercise what might be a good segue to

Couple of highlights:
* Linear SVC classifier is a good classification method
* Using heatmaps is a must - it helps to eliminate false positives

### Thoughts for improvements:
* Use more data for training for training the classifier
* Strengthen identification of vehicles via using cv2.matchTemplate template matching function
* Calculation of distances from the objects (e.g. vehicles) could help to set scale values to appropriate values. The implemented algorithm is not aware of how far or close is a given object and it makes identification harder. Knowing distances would also let to set ranges for sliding windows mechanism to define in a better way.

There is lots of hard-coded values like:
* Threshold for heatmap
* ranges for sliding windows
These items should be calculated in automated way.

The applied approach using linear SVC classifier is dependent on the sizes of objects. Because the ranges for sliding windows are hardcoded then they work properly assuming that objects are within appropriate distance from a car. Some automated way of sliding windows ranges and scale needs to be implemented so those values don't need to be hardcoded. Calculation of the lane of a car and neighboring lanes should be applied to define scale and ranges.

The classification mechanism should probably be implemented in some other language than Python - python processing might be too slow for real-time processing of images from car cameras.

One could combined the method used in this project with vehicle detection mechanism implemented based on neural networks - it would improve quality of object identifications.
