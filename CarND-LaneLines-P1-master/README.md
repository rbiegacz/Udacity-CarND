# **Finding Lane Lines on the Road** 
This project implements algorithm to finding lanes on the street while driving. It works both on static images as well on videos in the mpeg format.

Below you can find information about how the pipeline processing and annotating images was implementing and ideas for improving the implementation.

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The core function is process_image() function which implements the pipeline which consists of the following steps:
1. Changing image from color one to gray scale.
2. Selecting region of interest from the image. The shape of the selected region is a trapezoid.
3. Smoothing the image using Gaussian smoothing (https://en.wikipedia.org/wiki/Gaussian_blur) 
4. Selecting white color in the select region.
5. Discovering lines in the picture using Canny Edge Detection algorithm (https://en.wikipedia.org/wiki/Canny_edge_detector)
6. Drawing Hough lines.
7. Overlaying an image with drawn lines over the original image.

The examplary output of the pipeline looks like this: 

[//]: # (Image References)

[image1]: ./test_images/output_solidWhiteCurve.jpg "Annotated picture"

![alt text][image1]

draw_lines(...) function works in the following way.
* It goes thru all the lines identified by Canny Edge Detection alorithm.
* It calculates slope and y-intercept for each line detected by Canny Edge Detection algorithm.
* If absolute value of slope is smaller than 0.5 a line is rejected.
* Algorithm divides lines into two groups: one with lines with negative and one with lines with postitive slope.
* Algorithm tries to select the longest line out of the lines with positive slope.
  Once the line is selected, then based on slope and y-intercept values I calcualte coordinates of the points belonging to line 
  and lying at the bottom of the picture and and the top of the trapezoid.
* The previous step is repeated but for lines with negative slope.
* Finally, the lines are draw and image with the lines is returned by the function.

### 2. Identify potential shortcomings with your current pipeline
Current implementation of draw_lines() function picks only two lines: left and right line. That's why the lines are shaky which is visible in annotated videos. One could use a different mechanism for "selecting"/calculating lines which was suggested in the next section.

Anoher shortcoming is that lines are rejected based on slope. If absolute value of a line slope is smaller than 0.5 threshold then the line is rejected by draw_lines() function. 0.5 is ad hoc/arbitrary value - one could think about more sophistiated method of choosing a slope threshold value. 

### 3. Suggest possible improvements to your pipeline
draw_lines() algorithm can calculate lane lines based on average slope and average y-intercept - this "selection" mechanism probably will be better compared to selecing the longest line selected by Canny Edge Detection routine.
