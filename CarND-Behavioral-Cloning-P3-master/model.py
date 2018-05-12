"""
    This module implements model training for Udacity Behavioral Cloning Project
"""

#
# Copyright (c) 2018 Rafal Biegacz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import csv
import cv2
import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def training_history():
    history_object = model.fit_generator(train_generator)

RESIZE_SCALE = 4
y_size = 160
x_size = 320

DATA_PATH="data"

def generator(samples, batch_size=192):
    """
    this function returns data in batches - batch size is specified by batch_size parameters;
    using this function helps in situation when you don't have enough memory on your computer
    to load all the data into memory;
    this function uses the concept of generator (implemented based on 'yield')
    :param samples:
    :param batch_size: size of a batch to generate
    :return: batch of data consisting of features and expected result, size of it = batch_size
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.1
            for batch_sample in batch_samples:
                name = '{}/IMG/'.format(DATA_PATH)+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

                name = '{}/IMG/'.format(DATA_PATH)+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)

                name = '{}/IMG/'.format(DATA_PATH)+batch_sample[1].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def load_data(read_only_data=False):
    """
    this function is responsible for loading data and augmenting dataset
    :return:
    """
    lines = []
    with open('{}/driving_log.csv'.format(DATA_PATH)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    if len(lines) == 0:
        print("I cannot read data!")
    images = []
    measurements = []
    aug_images = []
    aug_measurements = []

    if read_only_data:
        return lines[1:]

    for line in lines[1:]:
        source_path = line[0]
        source_path_left = line[1]
        source_path_right = line[2]

        filename = source_path.split('/')[-1]
        filename_left = source_path_left.split('/')[-1]
        filename_right = source_path_right.split('/')[-1]

        current_path = '{}/IMG/'.format(DATA_PATH) + filename
        current_path_left = '{}/IMG/'.format(DATA_PATH) + filename_left
        current_path_right = '{}/IMG/'.format(DATA_PATH) + filename_right

        image = cv2.imread(current_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_left = cv2.imread(current_path_left)
        image_left=cv2.cvtColor(image_left,cv2.COLOR_BGR2RGB)
        image_right = cv2.imread(current_path_right)
        image_right=cv2.cvtColor(image_right,cv2.COLOR_BGR2RGB)

        steering_center = float(line[3])
        correction = 0.1
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        images.append(image)
        measurements.append(steering_center)

        aug_images.append(image)
        aug_measurements.append(steering_center)

        aug_images.append(np.fliplr(image))
        aug_measurements.append(-steering_center)

        aug_images.append(image_left)
        aug_measurements.append(steering_left)
        aug_images.append(np.fliplr(image_left))
        aug_measurements.append(-steering_left)

        aug_images.append(image_right)
        aug_measurements.append(steering_right)
        aug_images.append(np.fliplr(image_right))
        aug_measurements.append(-steering_right)
    return aug_images, aug_measurements

def create_model(model_type):
    models = {"simple", "lenet", "advanced"}
    model = None
    if not (model_type in models):
        print("Wrong model type!")
        exit()

    if model_type == "simple" :
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((40, 10), (0, 0))))
        model.add(Flatten())
        model.add(Dense(1))
    elif model_type == "advanced":
        # implementation of https://devblogs.nvidia.com/deep-learning-self-driving-cars/
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
        model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
        model.add(Convolution2D(64, 3, 3, activation="relu"))
        model.add(Convolution2D(64, 3, 3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(1))
    elif model_type == "lenet":
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((40, 10), (0, 0))))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        model.add(Dense(84))
        model.add(Dense(1))
    else:
        print("Wrong model type!")
    return model


def train_model(model_type):
    """
    this function is responsible for training the model
    :return:
    """
    aug_images, aug_measurements = load_data()

    X_train = np.array(aug_images)
    y_train = np.array(aug_measurements)

    models = {"simple", "lenet", "advanced"}
    if not (model_type in models):
        print("Wrong model type!")
        exit()

    model = create_model(model_type)
    if model is None:
        exit()

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    if model_type == "simple":
        model.save('t1_simple_model.h5')
    elif model_type == "advanced":
        model.save('t1_advanced_model.h5')
    else:
        model.save('t1_lenet_model.h5')

def train_model_2(model_type):
    """
    this function is responsible for training the model
    :return:
    """
    samples = load_data(read_only_data=True)
    print("Number of all samples: {}".format(6*len(samples)))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print("Number of all training samples: {}".format(6*len(train_samples)))
    print("Number of all validation samples: {}".format(6*len(validation_samples)))

    models = {"simple", "lenet", "advanced"}
    if not (model_type in models):
        print("Wrong model type!")
        exit()

    model = create_model(model_type)
    if model is None:
        exit()

    train_generator = generator(train_samples, batch_size=192)
    validation_generator = generator(validation_samples, batch_size=192)

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch= 6*len(train_samples),\
                        validation_data = validation_generator,\
                        nb_val_samples = 6*len(validation_samples)/192, nb_epoch = 5)

    if model_type == "simple":
        model.save('t2_simple_model.h5')
    elif model_type == "advanced":
        model.save('t2_advanced_model.h5')
    else:
        model.save('t2_lenet_model.h5')

if __name__ == '__main__':
    train_model_2("advanced")