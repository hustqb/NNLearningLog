# coding=utf-8
"""
keras.preprocessing.image.ImageDataGenerator
图像生成器，按批次实时的生成训练数据
"""
from __future__ import print_function
import sys
sys.path.append('/home/zqb/zqb_code/NNLearningLog/example')
from utils.datasets import load_cifar10
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator


def cnn(input_shape):
	input_data = Input(input_shape)
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_data)
	x = Conv2D(32, (3, 3), activation='relu')(x)
	x = MaxPool2D((2, 2))(x)
	x = Dropout(0.25)(x)

	x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
	x = Conv2D(64, (3, 3), activation='relu')(x)
	x = MaxPool2D((2, 2))(x)
	x = Dropout(0.25)(x)

	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(10, activation='softmax')(x)

	return Model(input_data, x)


(x_train, y_train), (x_test, y_test) = load_cifar10()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

datagen = ImageDataGenerator(
	featurewise_center=True,  # Set input mean to 0 over the dataset.
	featurewise_std_normalization=True,  # Divide inputs by std of the dataset.
	rotation_range=20,  # Degree range for random rotations.
	width_shift_range=0.2,  # Range(fracion of total width) for random horizontal shifts.
	height_shift_range=0.2,  # Range(fraction of total height) for random vertical shifts.
	horizontal_flip=True  # Randomly flip inputs horizontally.
)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# fit(x): Compute the internal data stats related to the data-dependent
# transformations, based on an array of sample data. Only required if
# "featurewise_center" or "featurewise_std_normalization" or "zca_whitening".
datagen.fit(x_train)

# get cnn model
model = cnn((32, 32, 3))
model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

# '''fits the model on batches with real-time data augmentation'''
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=len(x_train) / 32, epochs=100)

# '''here's a more "manual" example'''
# for epoch in range(100):
# 	print('Epoch=>', epoch)
# 	batches = 0
# 	# datagen.flow(x, y):
# 	# Takes numpy data & label arrays, and generate batches of augmented/normalized data.
# 	# Yields batches indefinitely, in an infinite loop.
# 	# yield: Tuples of (x, y) where x is numpy array of image data and y is a
# 	# numpy array of corresponding labels.
# 	for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
# 		model.fit(x_batch, y_batch)
# 		print(model.get_weights()[-1])
# 		batches += 1
# 		if batches >= len(x_train) / 32:
# 			# we need to break the loop by hand because
# 			# the generator loops indefinitely
# 			break
