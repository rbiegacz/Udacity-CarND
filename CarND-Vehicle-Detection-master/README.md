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

Summary of the used methods
---
### Data set and classification/identification method
Data set containing vehicles and non-vehicles consists of
- 8792  cars and 
- 8968  non-cars

Dataset exploration is implemented in object_detection_car_notcar.py file and is shown in vehicle_detection.ipynb in cells #6-#9
All the images are used in .png format to avoid problems related to different interpration of .jpg and .png data.

Linear Support Vector Classification (implemented within Scikit Learn library) is used to train classifier to detect vehicles. The whole training set is used to train the classifier.



###Histogram of Oriented Gradients (HOG)


###Sliding Window Search
The pictures taken by camera were devided into small pieces as it was shown in cell #

###Heatmap
Heatmap mechanism was used to identify areas detected as 'vehicles' by many boxes. This method allowed to eliminate a lot of false positives. 

###Video Processing
Video processing is implemented in the last cells of the Jupyter Notebook in the section called "Video Processing".

Processing is done frame by frame and the output is stored in the form of mp4 file.  


Summary and Improvement Suggestions
---
###Project Conclusions
Project is a good exploration exercise what might be a good segue to 

Couple of hightlights:
* Linear SVC classifier is a good classification method
* Using heatmaps is a must - it helps to eliminate false positives

### Thoughts for improvments:
* Use more data for training for training the classifier
* Strengthen identification of vehicles via using cv2.matchTemplate template matching function
* Calculation of distances from the objects (e.g. vehicles) could help to set scale values to appropriate values. The implemented algorithm is not aware of how far or close is a given object and it makes identification harder. Knowing distances would also let to set ranges for sliding windows mechanism to define in a better way.

There is lots of hard-coded values like:
* Threashold for heatmap
* ranges for sliding windows
These items should be calculated in automated way.

The applied approach using linear SVC classifier is depenent on the sizes of objects. Because the ranges for sliding windows are hardcoded then they work properly assuming that objects are within appropriate distance from a car. Some automated way of sliding windows ranges and scale needs to be implemented so those values don't need to be hardcoded. Calculation of the lane of a car and neighboring lanes should be applied to define scale and ranges.

The classification mechanism should probably be implemented in some other language than Python - python processing might be too slow for real-time processing of images from car cameras.

One could combined the method used in this project with vehicle detection mechanism implemented based on neural networks - it would improve quality of object identifications.


