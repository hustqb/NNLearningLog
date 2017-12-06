# coding=utf-8
"""
Trains a simple convnet on the MNIST dataset.
"""

from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from utils.datasets import load_mnist
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


img_shape = (28, 28, 1)
n_classes = 10
batch_size = 128
epochs = 2
outpath = 'mnist_first_output'


def cnn():
	input_data = Input(img_shape)
	x = Conv2D(32, (3, 3), activation='relu', name='conv_1')(input_data)
	x = Conv2D(64, (3, 3), activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(n_classes, activation='softmax')(x)

	model = Model(input_data, x)
	intermediate_model = Model(input_data, model.get_layer('conv_1').output)
	return model, intermediate_model


def preprocess(*arrays):
	x_train, y_train, x_test, y_test = arrays
	x_train = x_train.reshape((x_train.shape[0],) + img_shape)
	x_test = x_test.reshape((x_test.shape[0],) + img_shape)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	return x_train, y_train, x_test, y_test


def plot(inter, weight, org_data):
	"""
	:param inter: 第一层输出
	:param weight: 第一层权重
	:param org_data: 原始数据
	:return:
	"""
	if not os.path.exists(outpath):
		os.mkdir(outpath)

	for i in range(5):  # 绘制测试集中的前5个手写数字
		cur_inter = inter[i]
		cur_inter = cur_inter.reshape(cur_inter.shape[:3])
		cur_org = org_data[i]
		cur_org = cur_org.reshape(cur_org.shape[:2])

		'''第一层输出'''
		fig1, axes = plt.subplots(4, 8)
		axes = axes.flatten()
		for ax, j in zip(axes, range(cur_inter.shape[2])):
			ax.imshow(cur_inter[:, :, j])
		fig1.savefig(os.path.join(outpath, str(i) + 'inter.png'))
		plt.close(fig1)

		'''第一层输入'''
		fig2 = plt.figure()
		plt.imshow(cur_org)
		fig2.savefig(os.path.join(outpath, str(i) + 'org.png'))
		plt.close(fig2)

	'''第一层权重'''
	weight = weight.reshape(3, 3, 32)
	fig3, axes = plt.subplots(4, 8)
	axes = axes.flatten()
	for ax, j in zip(axes, range(weight.shape[2])):
		ax.imshow(weight[:, :, j])
	fig3.savefig(os.path.join(outpath, 'weight.png'))
	plt.close(fig3)


def main():
	(x_train, y_train), (x_test, y_test) = load_mnist()
	x_train, y_train, x_test, y_test = preprocess(x_train, y_train,
	                                              x_test, y_test)
	model, intermediate_model = cnn()
	model.compile(loss='categorical_crossentropy',
	              optimizer='sgd',
	              metrics=['accuracy'])
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	'''绘制中间层输出结果'''
	intermediate_output = intermediate_model.predict(x_test)
	intermediate_weight = intermediate_model.get_weights()
	plot(intermediate_output, intermediate_weight[0], x_test)


if __name__ == '__main__':
	main()
