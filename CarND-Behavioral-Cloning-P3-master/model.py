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

def generator(samples, batch_size=192):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.1
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

                name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)

                name = 'data/IMG/'+batch_sample[1].split('/')[-1]
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
    with open('data/driving_log.csv') as csvfile:
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

        current_path = 'data/IMG/' + filename
        current_path_left = 'data/IMG/' + filename_left
        current_path_right = 'data/IMG/' + filename_right

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

def train_model():
    """
    this function is responsible for training the model
    :return:
    """
    aug_images, aug_measurements = load_data()

    X_train = np.array(aug_images)
    y_train = np.array(aug_measurements)

    simple_model = False

    if simple_model:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((50, 10), (0, 0))))
        model.add(Flatten())
        model.add(Dense(1))
    else:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((40, 10), (0, 0))))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
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

        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

    if simple_model:
        model.save('simple_model.h5')
    else:
        model.save('lenet_model.h5')

def train_model_2():
    """
    this function is responsible for training the model
    :return:
    """
    samples = load_data(read_only_data=True)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    simple_model = False

    if simple_model:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((50, 10), (0, 0))))
        model.add(Flatten())
        model.add(Dense(1))
    else:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(y_size,x_size,3)))
        model.add(Cropping2D(cropping=((40, 10), (0, 0))))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
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

        train_generator = generator(train_samples, batch_size=192)
        validation_generator = generator(validation_samples, batch_size=192)

        model.compile(loss='mse', optimizer='adam')
        model.fit_generator(train_generator, samples_per_epoch=len(train_samples),\
                            validation_data = validation_generator,\
                            nb_val_samples = len(validation_samples), nb_epoch = 5)
    if simple_model:
        model.save('simple_model.h5')
    else:
        model.save('lenet_model.h5')

if __name__ == '__main__':
    train_model_2()