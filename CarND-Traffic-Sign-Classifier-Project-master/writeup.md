**Traffic Sign Recognition** - Project Report

---

**Project Purpose**

The goal of this project is to implement German road signs recognition based on deep neural networks (especially convolutional networks) with the use of Tensorflow. 

Project implementation will consist of the following steps:
* Loading the data set that will be used for training, validation and measuring algorithm accuracy
* Exploring, summarizing and visualizing the data set
* Designing, training and testing a model architecture
* Using the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image0]: dataset_characteristics.png "Dataset"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/sign1_32_32.png "Traffic Sign 1"
[image5]: ./test_images/sign2_32_32.png "Traffic Sign 2"
[image6]: ./test_images/sign3_32_32.png "Traffic Sign 3"
[image7]: ./test_images/sign4_32_32.png "Traffic Sign 4"
[image8]: ./test_images/sign5_32_32.png "Traffic Sign 5"
[image9]: ./test_images/sign6_32_32.png "Traffic Sign 6"
[image10]: ./test_images/sign7_32_32.png "Traffic Sign 7"
[image11]: ./test_images/sign8_32_32.png "Traffic Sign 8"
[image12]: ./test_images/sign9_32_32.png "Traffic Sign 9"
[image13]: ./test_images/sign10_32_32.png "Traffic Sign 10"
[image14]: ./test_images/sign10_32_32.png "Traffic Sign 11"

Project Quality Expectations
Project was implemented to meet the requirements mentioned in [rubric points](https://review.udacity.com/#!/rubrics/481/view).
Below you can find information about how each point mentioned there is addressed/answers in the project.

Here is the link to [project code](https://github.com/rbiegacz/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---
#### 1. Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:
34799
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 pixels x 32 pixels and there are 3 channels (it means that images are colorful)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

It is a bar chart showing how many pictures belonged to a specific road sign category (in total we have 43 types of signs)

![alt text][image0]

### Design and Test a Model Architecture

#### 1. Data preparationn
I decided to use normalized color pictures of road signs.

Here is an example of an original image and an augmented image:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image14]
![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
