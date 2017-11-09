# **Finding Lane Lines on the Road** 
This project implements algorithm to finding lanes on the street while driving. It works both on static images as well on videos in the mpeg format.

Below you can find information about how the pipeline processing and annotating images was implementing and ideas for improving the implementation.

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.


My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][test_images\output_solidWhiteCurve.jpg]

The following methods will be used to achieve this goal. Pictures will be processed (turned into gray color scheme, we will limit region of interesto to the secio

The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection. 



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
