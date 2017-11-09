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

![alt text][test_images\output_solidWhiteCurve.jpg]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
