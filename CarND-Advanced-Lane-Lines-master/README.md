## Advanced Line Lanes Discovery 
(project within Udacity Car NanoDegree)

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration2.jpg "Distorted"
[image1]: ./output_images/undist_calibration2.jpg "Undistorted"
[image2]: ./test_images/test3.jpg "Original image"
[image10]: ./output_images/undist_test3.jpg "Undistorted Image"
[image3]: ./output_images/lines_undist_test3.jpg "Binary Example"
[image4]: ./output_images/warped_lines_undist_test3_10percent.jpg "Warp Example"
[image5]: ./output_images/per_warped_lines_undist_test3_10percent.jpg "Fit Visual"
[image6]: ./output_images/annotated_test3.jpg "Output"
[video1]: ./annotated_project_video.mp4 "Video"

## Meeting expectations described on Udacity web pages in [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 

---
### Camera Calibration 

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function camera_calibration(lines 9 through 46) implemented in in correectcamera.py 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Orginal image (distorted one):
![alt text][image0]

Processed image (distortion removed):
![alt text][image1]


### Pipeline (single images)

#### 1. Removing distortion from camera images.

Based on camera calibration images (delivered in camera_cal folder) I calculate camera matrix and distortion coefficients using cv2.calibrateCamera function. Once I have camera matrix and distortion coefficients I can remove distortion from camera images (or from video frames that are processes as if they are regular images).

Here is original camera picture (example):

![alt text][image2]

Here is how it looks after distortion removal:

![alt text][image10]

#### 2. Identification of lines in images

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are implemented in thresholds_pipeline function in searchlines.py file). 

Here's an example of my output from this step.

![alt text][image3]

#### 3. Perspective Transformation to birds-eye view.

The code responsible for transformation of binary images into bird's eye view images is situated in perspective_transform(...) function (lines 23 thru 69 in perspectivetransform.py).

```python
src = TO BE DETERMINED
dst = TO BE DETERMINED
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane identification using polynomials

The code responsible for calculating formulas for lines is in the function '''python search_for_lines()''' in seachlines.py file. 

An example of an image with line detected is presented below.

![alt text][image5]

#### 5. Calculation the radius of curvature of the lane and the position of the vehicle with respect to center.

Calculation of the radius of the curvature is implemented in determine_lane_curvature function (lines 150 thru 182) in roadlanes.py file.

#### 6. Plotting Lane Road

Processing single static images: in case of static image I use draw_lane(...) function (lines 21 thru 49) from roadlane.py file - this function draws a lane. Further I add information about radius of curvature of the lane - I retrieve this information from determine_lane_curvature(...) function. 

![alt text][image6]

---

### Pipeline (video)

#### 1. Annotated project video  

The pipeline for annotating the video was implemented in annotate_movie(...) function which is in lines 184 thru 245 in roadlanes.py.

Here's a [link to my video result](./annoated_project_video.mp4)

---

### Discussion

The current state of the project is a good start for more fune tunning.

For example, to generate binary images in apply_gradients_thresholds function  I used only SobelX operator and used S channel from HLS color space. One can definitely, improve the function using magnitude of the gradient, direction of the gradient.

Performance improvements: right now, processing a single image/frame from the first step until annotation takes ~1s. This performance should be improved. For example, using convolution in sliding window search would accelerate processing.


