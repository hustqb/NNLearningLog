# coding=utf-8
"""
Transfer learning toy example.
1 - Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
2 - Freeze convolutional layers and fine-tune dense layers
   for the classification of digits [5..9].
Get to 99.8% test accuracy after 5 epochs
for the first five digits classifier
and 99.2% for the last five digits after transfer + fine-tuning.
"""

from __future__ import print_function

import datetime
import keras
from utils.datasets import load_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

now = datetime.datetime.now


def preprocess(train, test):
	x_train = train[0].reshape((train[0].shape[0],) + (28, 28, 1))
	x_test = test[0].reshape((test[0].shape[0],) + (28, 28, 1))
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(train[1], 5)
	y_test = keras.utils.to_categorical(test[1], 5)
	return x_train, x_test, y_train, y_test


def train_model(model, train, test):
	"""
	:param model:
	:param train: 训练集
	:param test: 测试集
	:return:
	"""
	x_train, x_test, y_train, y_test = preprocess(train, test)

	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])

	t = now()
	model.fit(x_train, y_train,
	          batch_size=128,
	          epochs=5,
	          verbose=1,
	          validation_data=(x_test, y_test))
	print('Training time: %s' % (now() - t))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


def mnist_transfer_cnn():
	(x_train, y_train), (x_test, y_test) = load_mnist()

	# create two datasets: one with digits < 5 and one >= 5
	x_train_lt5 = x_train[y_train < 5]
	y_train_lt5 = y_train[y_train < 5]
	x_test_lt5 = x_test[y_test < 5]
	y_test_lt5 = y_test[y_test < 5]
	x_train_gte5 = x_train[y_train >= 5]
	y_train_gte5 = y_train[y_train >= 5] - 5
	x_test_gte5 = x_test[y_test >= 5]
	y_test_gte5 = y_test[y_test >= 5] - 5

	# define two groups of layers:
	# feature (convolutions) and classification (dense)
	feature_layers = [
		Conv2D(32, 3, padding='valid',
		       input_shape=(28, 28, 1),
		       activation='relu'),
		Conv2D(32, 3, activation='relu'),
		MaxPooling2D(2),
		Dropout(0.25),
		Flatten()
	]
	classification_layers = [
		Dense(128, activation='relu'),
		Dropout(0.5),
		Dense(5, activation='softmax')
	]

	# create complete model
	# 这是keras中Sequential的一个神奇的用法，它是一个stack
	model = Sequential(feature_layers + classification_layers)

	# train model for 5-digit classification 0-4
	train_model(model,
	            (x_train_lt5, y_train_lt5),
	            (x_test_lt5, y_test_lt5))

	# freeze feature layers and rebuild model
	for l in feature_layers:
		l.trainable = False

	# transfer: train dense layers for new classification task [5..9]
	train_model(model,
	            (x_train_gte5, y_train_gte5),
	            (x_test_gte5, y_test_gte5))
