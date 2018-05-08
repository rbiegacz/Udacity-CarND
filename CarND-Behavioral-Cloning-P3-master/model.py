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

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def load_data():
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

    for line in lines[1:]:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        aug_images.append(np.fliplr(image))
        measurements.append(float(line[3]))
        aug_measurements.append(-float(line[3]))
    return aug_images, aug_measurements

def train_model():
    """
    this function is responsible for training the model
    :return:
    """
    aug_images, aug_measurements = load_data()

    X_train = np.array(aug_images)
    y_train = np.array(aug_measurements)

    simple_model = True

    if simple_model:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        model.add(Flatten(input_shape=(160, 320, 3)))
        model.add(Dense(1))
    else:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

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
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

    if simple_model:
        model.save('simple_model.h5')
    else:
        model.save('lenet_model.h5')

if __name__ == '__main__':
    train_model()