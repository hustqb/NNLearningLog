# coding=utf-8
"""
siamese网络
"""
import os
import random
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.layers import Lambda
from keras.models import Model


def load_mnist():
	path = os.path.join('/home/zqb/zqb_code/NNLearningLog', 'famousData/mnist.npz')
	data = np.load(path)
	return (data['x_train'], data['y_train']), \
	       (data['x_test'], data['y_test'])


def preprocess(x_train, x_test):
	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	return x_train, x_test


def create_pairs(x, digit_indices):
	"""Positive and negtive pair creation.
	Alternates between positive and negative pairs."""
	pairs = []
	labels = []
	n = min([len(digit_indices[d]) for d in range(10)]) - 1
	for d in range(10):
		# 对第d类抽取正负样本
		for i in range(n):
			# 遍历d类的样本，取临近的两个样本为正样本对
			z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
			pairs += [[x[z1], x[z2]]]
			# randrange会产生1~9之间的随机数，含1和9
			inc = random.randrange(1, 10)
			# (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
			dn = (d + inc) % 10
			# 在d类和dn类中分别取i样本构成负样本对
			z1, z2 = digit_indices[d][i], digit_indices[dn][i]
			pairs += [[x[z1], x[z2]]]
			# 添加正负样本标签
			labels += [1, 0]
	return np.array(pairs), np.array(labels)


def create_base_nn(input_dim):
	"""Base network to be shared (eq. to feature extraction"""
	input_data = Input((input_dim,))
	x = Dense(128, activation='relu')(input_data)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	model = Model(input_data, x)
	return model


def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
	"""Contrastive loss"""
	margin = 1
	return K.mean(y_true * K.square(y_pred) +
	              (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_model(base_nn, input_dim):
	input_a = Input((input_dim,))
	input_b = Input((input_dim,))
	processed_a = base_nn(input_a)
	processed_b = base_nn(input_b)
	distance = Lambda(euclidean_distance,
	                  output_shape=eucl_dist_output_shape)([processed_a,
	                                                        processed_b])
	model = Model([input_a, input_b], distance)
	return model


def compute_accuracy(predictions, labels):
	"""Compute classification accuracy with a fixed threshold on distances.
	"""
	return labels[predictions.ravel() < 0.5].mean()


def main():
	(x_train, y_train), (x_test, y_test) = load_mnist()
	x_train, x_test = preprocess(x_train, x_test)
	digit_indices = [np.where(y_train == i)[0] for i in range(10)]
	tr_pairs, tr_y = create_pairs(x_train, digit_indices)
	digit_indices = [np.where(y_test == i)[0] for i in range(10)]
	te_pairs, te_y = create_pairs(x_test, digit_indices)

	base_nn = create_base_nn(input_dim=784)
	model = create_model(base_nn, input_dim=784)

	rms = keras.optimizers.RMSprop()
	model.compile(loss=contrastive_loss, optimizer=rms)
	model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
	          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
	          batch_size=128, nb_epoch=20)
	pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
	tr_acc = compute_accuracy(pred, tr_y)
	pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
	te_acc = compute_accuracy(pred, te_y)

	print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
	print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


if __name__ == '__main__':
	main()
