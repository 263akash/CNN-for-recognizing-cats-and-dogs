# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:56:16 2019

@author: Administrator
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,      # 1.we rescaled all our pixel value b/w 0 and 1(because pixel gets value from 0 t 255)
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'D:\\ML\\Learning\Machine Learning A-Z Template Folder\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks\\dataset\\training_set',
        target_size=(64, 64),  #target size is the size of images expected in our CNN model
        batch_size=32,
        class_mode='binary')  #we have two class cat and dogs (so binary))

test_set = test_datagen.flow_from_directory(
        'D:\\ML\Learning\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks\\dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000)
