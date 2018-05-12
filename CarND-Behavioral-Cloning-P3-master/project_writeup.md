**Behavioral Cloning Project**

The goal of this project is to use neural network to steer a car so it runs thrughout the whole race track without human intervantion.

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

This project fulfills requirements/expectations that are documented in [rubric points](https://review.udacity.com/#!/rubrics/432/view).

---
### Files Submitted & Code Quality

#### 1. Project structure and project artifacts

My project includes the following files:
* model.py containing the script to create and train the model
  the following functions in model.py are key for the whole project:
  a) load_data(...) - this function loads the information about training data from driving_log.csv
  b) generator(...) - generates a batch of (X, y) data of batch_size that is used to train a model; this function allows to read data in chunks and helps especially in case you don't have enough computer memory to load all the training data at once.
  c) create_model(...) this function creates a topology of a model - there are 3 types of models created by it:
  - "simple" (just for test purposes)
  - "sophisticated" - based on LeNet topology
  - "advanced" - based on NN topology described in this article: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    (normalization layer, cropping layer, 5 convolutional layers, 4 fully connected layers)
  
* drive.py for driving the car in autonomous mode
  this file serves data to car simulator and allows for capturing images from autonomous drive of a car

* model.h5 containing a trained convolution neural network
  this is the trained model used futher in autonomous mode to drive the car and record project video
  
* writeup_report.md or writeup_report.pdf summarizing the results
  this is the file that you're reading

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy



#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final architecture of the neural network used for this project is similar to what it was described in NVIDIA article: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

The network topology consists of the following layers:
1. Normalization layer (model.py: 178) using Keras lambda layer.
2. Cropping the image only to this part of the image that is relevant for making steering decisions (model.py: 179)
3. Convolutional Layer #1 (model.py: )
   the output of this layer goes thru RELU activation function to introduce non-linearity
4. Convolutional Layer #2 (model.py: )
5. Convolutional Layer #3 (model.py: )
6. Convolutional Layer #4 (model.py: )
7. Convolutional Layer #5 (model.py: )
8. Fully connected layer (model.py: 186)
9. Fully connected layer (model.py: 187)
9. Fully connected layer (model.py: 188)
9. Fully connected layer (model.py: 189)

#### 3. Attempts to reduce overfitting in the model

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 273).


The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py: lines 270-271). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Creation of the Training Set & Training Process


To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
