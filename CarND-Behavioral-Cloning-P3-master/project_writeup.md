**Behavioral Cloning Project**

The goal of this project is to use neural network to steer a car so it runs thrughout the whole race track without human intervantion.

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_camera.jpg "Center Camera"
[image2]: ./examples/left_camera.jpg "Left Camera"
[image3]: ./examples/right_camera.jpg "Right Image"
[image4]: ./examples/right_edge.jpg "Recovery - right side"
[image5]: ./examples/right_edge_half_recovery.jpg "Half recovery - right side"
[image6]: ./examples/right_edge_recovery.jpg "Full recovery - right side"

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
  - "lenet" - based on LeNet topology
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

### Training Set

When it comes to training set I recorded the following:
a) 3 full laps
b) 1 full lap driven in an opposite direction
c) Recovery from a situation that a car is on the right edge of the road
d) Recovery from a situation that a car is on the left edge of the road

During the learning process I take into account images from centeral, right and left camera (as it is presented below):
![Center Camera][image1]
![Right Camera][image2]
![Left Camera][image3]


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to come up with a neural network topology that is complex/sophisticated enough to produce correct steering actions to a car during inference phase.

I initially started with LeNet topology (the trained model is stored as t2_lenet_model.h5). Unfortunately, this topology was not enough to achive project goal - the car managed to drive thru two curves and then fell off the track.

As the next step I decided to follow guidance of NVIDIA engineers and used the topology that they presented in the following article: https://devblogs.nvidia.com/deep-learning-self-driving-cars/. The topology presented in this article allowed for achiving the project goal.

#### 2. Final Model Architecture

The final architecture of the neural network used for this project is similar to what it was described in NVIDIA article: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

The network topology consists of the following layers:
1. Normalization layer (model.py: 178) using Keras lambda layer.
2. Cropping the image only to this part of the image that is relevant for making steering decisions (model.py: 179)
3. Convolutional Layer #1 (model.py: line 181)
   model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
   the output of this layer goes thru RELU activation function to introduce non-linearity
4. Convolutional Layer #2 (model.py: line 182)
   model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
   the output of this layer goes thru RELU activation function to introduce non-linearity
5. Convolutional Layer #3 (model.py: line 183)
   model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
   the output of this layer goes thru RELU activation function to introduce non-linearity
6. Convolutional Layer #4 (model.py: line 185)
   model.add(Convolution2D(64, 3, 3, activation="relu"))
   the output of this layer goes thru RELU activation function to introduce non-linearity
7. Convolutional Layer #5 (model.py: line 186)
   model.add(Convolution2D(64, 3, 3, activation="relu"))
   the output of this layer goes thru RELU activation function to introduce non-linearity
8. Fully connected layer (model.py: line 187)
9. Fully connected layer (model.py: line 188)
9. Fully connected layer (model.py: line 189)
9. Fully connected layer (model.py: line 190)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 274).

#### 4. Training Process

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py: lines 271-272). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I managed to use course-provided data, I didn't need to prepare additional data to train models.

Training data consited of images captured by center, left and right camera and I also flipped images & angles to augment the training data.

I finally randomly shuffled the data set and put Y% of the data into a validation set (model.py: line 58 in generator(...) function).

The console output from the training sessions is presented below:

$ python model.py 
Using TensorFlow backend.
Train on 38572 samples, validate on 9644 samples
Epoch 1/5
38572/38572 [==============================] - 212s - loss: 0.0115 - val_loss: 0.0107
Epoch 2/5
38572/38572 [==============================] - 214s - loss: 0.0099 - val_loss: 0.0111
Epoch 3/5
38572/38572 [==============================] - 213s - loss: 0.0094 - val_loss: 0.0110
Epoch 4/5
38572/38572 [==============================] - 214s - loss: 0.0090 - val_loss: 0.0124
Epoch 5/5
38572/38572 [==============================] - 215s - loss: 0.0087 - val_loss: 0.0119

As you can see 3-5 epochs were enought to get fully functional/trained model that allowed to achieve project goals.

#### Results

The video that captures that ride of a car steered by neural network-based model is stored in video.mp4 file.
