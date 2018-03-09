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
[image14]: ./test_images/sign11_32_32.png "Traffic Sign 11"

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

There are following road signs in the dataset:

0 : Speed limit (20km/h)

1 : Speed limit (30km/h)

2 : Speed limit (50km/h)

3 : Speed limit (60km/h)

4 : Speed limit (70km/h)

5 : Speed limit (80km/h)

6 : End of speed limit (80km/h)

7 : Speed limit (100km/h)

8 : Speed limit (120km/h)

9 : No passing

10 : No passing for vehicles over 3.5 metric tons

11 : Right-of-way at the next intersection

12 : Priority road

13 : Yield

14 : Stop

15 : No vehicles

16 : Vehicles over 3.5 metric tons prohibited

17 : No entry

18 : General caution

19 : Dangerous curve to the left

20 : Dangerous curve to the right

21 : Double curve

22 : Bumpy road

23 : Slippery road

24 : Road narrows on the right

25 : Road work

26 : Traffic signals

27 : Pedestrians

28 : Children crossing

29 : Bicycles crossing

30 : Beware of ice/snow

31 : Wild animals crossing

32 : End of all speed and passing limits

33 : Turn right ahead

34 : Turn left ahead

35 : Ahead only

36 : Go straight or right

37 : Go straight or left

38 : Keep right

39 : Keep left

40 : Roundabout mandatory

41 : End of no passing

42 : End of no passing by vehicles over 3.5 metric tons

### Design and Test a Model Architecture

#### 1. Data preparation
I decided to use normalized color pictures of road signs so all pixel values for Red, Green and Blue are from the range [0, 1]

I decided not to convert to gray scale. Reduction to gray scale would allow to build smaller (from weights perspective) network that could be tought faster. The speed of learning wasn't an issue, though, in this particular case.

#### My final model consisted of the following layers:
| Layer         		|     Details	        					| Output |
|:----------------:|:--------:|:--------:| 
| Input         		| n/a | 32x32x3 RGB image |
| Convolutional | filter 6 with 1x1 stride and valid padding | outputs (?, 27, 27, 64) |
| Max Pooling | with stride 2x2 | outputs (?, 13, 13, 64) |
| Convolutional | filter 6 with 1x1 stride and valid padding | outputs (?, 8, 8, 64) |
| Max Pooling | with 2x2 stride | outputs (?, 4, 4, 64) |
| Flattening | n/a | outputs (?, 1024) |
| Fully Connected (Dense) | n/a | outputs (?, 688) |
| Fully Connected (Dense) | n/a | outputs (?, 344) |
| Dropout | dropout rate 0.7 | outputs (?, 344) |
| Fully Connected (Dense)| n/a | outputs (?, 86) |
| Dropout | dropout rate 0.8 | outputs (?, 86) |
| Logits | n/a | outputs (?, 43) |


#### 3. Hyperparameters used to achieve a model with accuracy higher than 95#

1. Batch size: 100 pictures

2. Learning rate: 0.0005

3. Number of Epochs: 50

4. Optimizer: [Adaptive Moment Estimation](http://ruder.io/optimizing-gradient-descent/index.html#adam) was used while training the network

#### 4. Final model and steps that led to it

To find the best model I did experimentation around:

1. Different sizes of filters used by convolutional networks. The best results were achieved when filter size was 5 or 6.

2. Different values of learning rate was tested from the range 0.001 to 0.0001. The value 0.005 seem to give the best results.

4. When it comes to number of epochs - in all cases, running more than 50 epochs didn't give better results then results achieved within first 50 epochs - that's why finally the maximum number of epochs used while learning was set to 50.

5. During the training, the algorithm saves the model paramters that give the best accuracy for validation set. It can happen in an epoch between epoch #1 and epoc #50. An example of training session could be seen below. 

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
 

#### 5. Final Model Result

My final model results were:
* validation set accuracy of 96.7%
* test set accuracy of 94%

Training...

EPOCH 1 ...
Validation Accuracy = 0.779
Model saved for accuracy: 0.7791383235362652


EPOCH 2 ...
Validation Accuracy = 0.848
Model saved for accuracy: 0.8482993245124817


EPOCH 3 ...
Validation Accuracy = 0.900
Model saved for accuracy: 0.8995464891533193


EPOCH 4 ...
Validation Accuracy = 0.904
Model saved for accuracy: 0.9043083920770761


EPOCH 5 ...
Validation Accuracy = 0.918
Model saved for accuracy: 0.9176870762896375


EPOCH 6 ...
Validation Accuracy = 0.920
Model saved for accuracy: 0.9195011348681115


EPOCH 7 ...
Validation Accuracy = 0.925
Model saved for accuracy: 0.925170073163212


EPOCH 8 ...
Validation Accuracy = 0.929
Model saved for accuracy: 0.9290249447703632


EPOCH 9 ...
Validation Accuracy = 0.921


EPOCH 10 ...
Validation Accuracy = 0.949
Model saved for accuracy: 0.9487528414412691


EPOCH 11 ...
Validation Accuracy = 0.937


EPOCH 12 ...
Validation Accuracy = 0.949


EPOCH 13 ...
Validation Accuracy = 0.940


EPOCH 14 ...
Validation Accuracy = 0.944


EPOCH 15 ...
Validation Accuracy = 0.943


EPOCH 16 ...
Validation Accuracy = 0.949


EPOCH 17 ...
Validation Accuracy = 0.946


EPOCH 18 ...
Validation Accuracy = 0.940


EPOCH 19 ...
Validation Accuracy = 0.939


EPOCH 20 ...
Validation Accuracy = 0.945


EPOCH 21 ...
Validation Accuracy = 0.943


EPOCH 22 ...
Validation Accuracy = 0.946


EPOCH 23 ...
Validation Accuracy = 0.947


EPOCH 24 ...
Validation Accuracy = 0.954
Model saved for accuracy: 0.9541950171766909


EPOCH 25 ...
Validation Accuracy = 0.942


EPOCH 26 ...
Validation Accuracy = 0.948


EPOCH 27 ...
Validation Accuracy = 0.953


EPOCH 28 ...
Validation Accuracy = 0.947


EPOCH 29 ...
Validation Accuracy = 0.945


EPOCH 30 ...
Validation Accuracy = 0.946


EPOCH 31 ...
Validation Accuracy = 0.959
Model saved for accuracy: 0.958730162947086


EPOCH 32 ...
Validation Accuracy = 0.951


EPOCH 33 ...
Validation Accuracy = 0.950


EPOCH 34 ...
Validation Accuracy = 0.948


EPOCH 35 ...
Validation Accuracy = 0.945


EPOCH 36 ...
Validation Accuracy = 0.922


EPOCH 37 ...
Validation Accuracy = 0.959
Model saved for accuracy: 0.9591836786053889


EPOCH 38 ...
Validation Accuracy = 0.960
Model saved for accuracy: 0.9596371915605333


EPOCH 39 ...
Validation Accuracy = 0.950


EPOCH 40 ...
Validation Accuracy = 0.928


EPOCH 41 ...
Validation Accuracy = 0.951


EPOCH 42 ...
Validation Accuracy = 0.957


EPOCH 43 ...
Validation Accuracy = 0.956


EPOCH 44 ...
Validation Accuracy = 0.960
Model saved for accuracy: 0.9600907099219947


EPOCH 45 ...
Validation Accuracy = 0.959


EPOCH 46 ...
Validation Accuracy = 0.952


EPOCH 47 ...
Validation Accuracy = 0.947


EPOCH 48 ...
Validation Accuracy = 0.954


EPOCH 49 ...
Validation Accuracy = 0.967
Model saved for accuracy: 0.9673469455874696


EPOCH 50 ...
Validation Accuracy = 0.956

INFO:tensorflow:Restoring parameters from ./lenet
Test Accuracy = 0.940

### Testing Model on New Images

#### 11 images of German traffic signs were downloaded from Internet. One can see them below.

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14]

The road signs presented on real pictures were much simpler to recognize by the network. The icons of road signs were much more difficult for the model to recognize - the main reason is that such examples of road signs were not part of training/validation/testing set.

#### 2. Here are the predictions for test images delivered by the developed model.

Here are the results of the prediction:

| Image			        |     Prediction	        					|  Correct in Top#5 |
|:---------------------:|:---------------------------------------:|:------:| 
| Speed limit (50km/h)      		| Speed limit (50km/h)   									|  #1 |
| No passing     			| No passing 										| #1 |
| Road work					| Road work											| #1 |
| Road narrows on the right      		| General caution					 				| no |
| Double curve			| Double curve      							| #1 |
| Speed limit (70km/h)      		| Dangerous curve to the left | #2 | 
| Yield     			| Yield 										| #1 |
| No Entry				| Priority road | #5 |
| Stop      		| No entry					 				| #2 |
| Roundabout mandatory			| Roundabout mandatory | #1 |
| Speed limit (70km/h)      		| Speed limit (30km/h) | no |


The model was able to correctly guess 6 out of 11 traffic signs, which gives an accuracy of 63.6%. This is worse compared to accuracy achieved for test test. 

What's interesting - if we take into account Top#5 labels for each sign then 9 answers out of 11 would be correct. Which is a little bit closer to accuracy for test set.

#### 3. Below you can find 5 highest probable lables predicted by the model for test images.

![alt text][image4]

Expected label for a sign #0: Speed limit (50km/h)

Selected label: Speed limit (50km/h) [CORRECT]

Predicted labels: ['Speed limit (50km/h)', 'Speed limit (30km/h)', 'Speed limit (80km/h)', 'Speed limit (60km/h)', 'Turn right ahead']

SoftMax Probabilities:  0.881826 0.118121 0.000053 0.000000 0.000000


IMAGE 1: as you can see the model categorized this road sign incorrectly - "Speed limit (30km/h)" instead of "Speed limit (50km/h)". The correct label, though, is pointed by the model as the second highest option.

![alt text][image5]

Expected label for a sign #1: No passing

Selected label: No passing [CORRECT]

Predicted labels: ['No passing', 'No passing for vehicles over 3.5 metric tons', 'Slippery road', 'Speed limit (60km/h)', 'Ahead only']

SoftMax Probabilities:  1.000000 0.000000 0.000000 0.000000 0.000000

IMAGE 2: based on softmax prob, model is pretty sure that it is right in this particular case.


![alt text][image6]

Expected label for a sign #2: Road work

Selected label: Road work [CORRECT]

Predicted labels: ['Road work', 'End of speed limit (80km/h)', 'General caution', 'Speed limit (30km/h)', 'End of no passing']

SoftMax Probabilities:  1.000000 0.000000 0.000000 0.000000 0.000000

IMAGE 3: based on softmax prob, model is pretty sure that it is right in this particular case.


![alt text][image7]

Expected label for a sign #3: Road narrows on the right

Selected label: General caution [INCORRECT]

Predicted labels: ['General caution', 'Speed limit (30km/h)', 'Speed limit (70km/h)', 'No entry', 'Traffic signals']

SoftMax Probabilities:  1.000000 0.000000 0.000000 0.000000 0.000000

IMAGE 4: based on softmax prob, model incorrect doesn't show a correct label with the 5 highest ones.


![alt text][image8]

Expected label for a sign #4: Double curve

Selected label: Double curve [CORRECT]

Predicted labels: ['Double curve', 'Beware of ice/snow', 'Right-of-way at the next intersection', 'Road work', 'Road narrows on the right']

SoftMax Probabilities:  0.999042 0.000787 0.000144 0.000016 0.000007

IMAGE 5: based on softmax prob, model is pretty sure that it is right in this particular case.


![alt text][image9]

Expected label for a sign #5: Speed limit (70km/h)

Selected label: Dangerous curve to the left [INCORRECT]

Predicted labels: ['Dangerous curve to the left', 'Speed limit (70km/h)', 'No passing', 'Traffic signals', 'Turn right ahead']

SoftMax Probabilities:  0.772409 0.156971 0.070562 0.000058 0.000000

IMAGE 6: based on softmax prob, model is pretty sure about "Dangerous curve to the left" label. The correct label is on the second position.


![alt text][image10]

Expected label for a sign #6: Yield

Selected label: Yield [CORRECT]

Predicted labels: ['Yield', 'No vehicles', 'Bumpy road', 'Stop', 'Keep left']

SoftMax Probabilities:  1.000000 0.000000 0.000000 0.000000 0.000000

IMAGE 7: based on softmax prob, model is pretty sure that it is right in this particular case.


![alt text][image11]

Expected label for a sign #7: No entry

Selected label: Priority road [INCORRECT]

Predicted labels: ['Priority road', 'Yield', 'Stop', 'No vehicles', 'No entry']

SoftMax Probabilities:  1.000000 0.000000 0.000000 0.000000 0.000000

Image 8: Model incorrectly classified this road sign. The correct answer in on the list of top #5 labels.

![alt text][image12]

Expected label for a sign #8: Stop

Selected label: No entry [INCORRECT]

Predicted labels: ['No entry', 'Stop', 'End of speed limit (80km/h)', 'Priority road', 'Keep right']

SoftMax Probabilities: 1.000000 0.000000 0.000000 0.000000 0.000000

Image 9: Model incorrectly classified this road sign. The correct answer in mentioned as 2nd option on the top #5 labels.

![alt text][image13]

Expected label for a sign #9: Roundabout mandatory

Selected label: Roundabout mandatory [CORRECT]

Predicted labels: ['Roundabout mandatory', 'Beware of ice/snow', 'Keep left', 'Go straight or left', 'General caution']

SoftMax Probabilities: 1.000000 0.000000 0.000000 0.000000 0.000000

IMAGE 10: based on softmax prob, model is pretty sure that it is right in this particular case.


![alt text][image14]

Expected label for a sign #10: Speed limit (70km/h)

Selected label: Speed limit (30km/h) [INCORRECT]

Predicted labels: ['Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Keep right', 'Yield']

SoftMax Probabilities: 0.976876 0.023124 0.000000 0.000000 0.000000


IMAGE 11: in this particular case, based on softmax probabilities, the model is pretty sure about selected label. The correct label is not within Top#5



#### 4. Summary

The designed model works OK for the training dataset. Its accuracy is higher than 93%.

For randomly selected road sign pictures its accuracy is around 63%.

The reason for that is that the data used to train the model doesn't include all the possible pictures. As the next step I would augment the training data with the pictures for which model doesn't perform as expected.



