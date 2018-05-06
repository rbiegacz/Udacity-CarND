# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The main goal in this project is to write a software pipeline to detect vehicles present in videos (test_video.mp4 and project_video.mp4).

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run processing pipeline on a video stream to detect vehicles in video frames 
* Estimate a bounding box around detected vehicles.

Solution Overview
---
This README is accompanied with Jupyter Notebook and a set of Python files that are realization of this project.

Name of Jupyter Notedbook is vehicle_detection.ipynb.

The main function responsible for processing of video frames is called object_detection_search_and_classify.py

Histogram of Oriented Gradients (HOG)
---


Sliding Window Search
---

Video Processing
---
Video processing is implemented in the last cells of the Jupyter Notebook - 

Summary and Improvement Suggestions
---
Project is a good exploration exercise what might be a good segue to 

Couple of hightlights:
* Linear SVC classifier is a good classification method


Thoughts for improvments:
* Use more data for training for training the classifier
* Strengthen identification of vehicles via using cv2.matchTemplate template matching function

There is lots of hard-coded values like:
* Threashold for heatmap
* ranges for sliding windows
These items should be calculated in automated way.

The applied approach using linear SVC classifier is depenent on the sizes of objects. Because the ranges for sliding windows are hardcoded then they work properly assuming that objects are within appropriate distance from a car. Some automated way of sliding windows ranges and scale needs to be implemented so those values don't need to be hardcoded. Calculation of the lane of a car and neighboring lanes should be applied to define scale and ranges.





