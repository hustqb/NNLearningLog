# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
"""
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)


def build_generator(latent_size):
	# we will map a pair of (z, L), where z is a latent vector and L is a
	# label drawn from P_c, to image space (..., 28, 28, 1)
	cnn = Sequential()

	cnn.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
	cnn.add(Reshape((3, 3, 384)))

	# upsample to (7, 7, ...)
	cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
	                        activation='relu',
	                        kernel_initializer='glorot_normal'))

	# upsample to (14, 14, ...)
	cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
	                        activation='relu',
	                        kernel_initializer='glorot_normal'))

	# upsample to (28, 28, ...)
	cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
	                        activation='tanh',
	                        kernel_initializer='glorot_normal'))

	# this is the z space commonly referred to in GAN papers
	latent = Input(shape=(latent_size,))

	# this will be our label
	image_class = Input(shape=(1,), dtype='int32')

	cls = Flatten()(Embedding(10, latent_size,
	                          embeddings_initializer='glorot_normal')(image_class))

	# hadamard product between z-space and a class conditional embedding
	h = layers.multiply([latent, cls])

	fake_image = cnn(h)

	return Model([latent, image_class], fake_image)


def build_discriminator():
	# build a relatively standard conv net, with LeakyReLUs as suggested in
	# the reference paper
	cnn = Sequential()

	cnn.add(Conv2D(32, 3, padding='same', strides=2,
	               input_shape=(28, 28, 1)))
	cnn.add(LeakyReLU(0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Conv2D(64, 3, padding='same', strides=1))
	cnn.add(LeakyReLU(0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Conv2D(128, 3, padding='same', strides=2))
	cnn.add(LeakyReLU(0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Conv2D(256, 3, padding='same', strides=1))
	cnn.add(LeakyReLU(0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Flatten())

	image = Input(shape=(28, 28, 1))

	features = cnn(image)

	# first output (name=generation) is whether or not the discriminator
	# thinks the image that is being shown is fake, and the second output
	# (name=auxiliary) is the class that the discriminator thinks the image
	# belongs to.
	fake = Dense(1, activation='sigmoid', name='generation')(features)
	aux = Dense(10, activation='softmax', name='auxiliary')(features)

	return Model(image, [fake, aux])


if __name__ == '__main__':
	# build the discriminator
	print('Discriminator model:')
	discriminator = build_discriminator()
	discriminator.compile(
		optimizer=Adam(lr=0.0002, beta_1=0.5),
		loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
	)
	discriminator.summary()

	# build the generator
	generator = build_generator(100)

	latent = Input(shape=(100,))
	image_class = Input(shape=(1,), dtype='int32')

	# get a fake image
	fake = generator([latent, image_class])  # a tensor

	# we only want to be able to train generation for the combined model
	discriminator.trainable = False
	fake, aux = discriminator(fake)  # two tensor
	combined = Model([latent, image_class], [fake, aux])

	print('Combined model:')
	combined.compile(
		optimizer=Adam(lr=0.0002, beta_1=0.5),
		loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
	)
	combined.summary()

	# get our mnist data, and force it to be of shape (..., 28, 28, 1)
	# range [-1, 1]
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.astype(np.float32) - 127.5) / 127.5
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = (x_test.astype(np.float32) - 127.5) / 127.5
	x_test = np.expand_dims(x_test, axis=-1)

	num_train, num_test = x_train.shape[0], x_test.shape[0]

	train_history = defaultdict(list)
	test_history = defaultdict(list)

	for epoch in range(1, 50 + 1):
		print('Epoch {}/{}'.format(epoch, 50))

		num_batches = int(x_train.shape[0] / 100)
		progress_bar = Progbar(target=num_batches)

		# we don't want the discriminator to also maximize the classification
		# accuracy of the auxiliary classifier on generated images, so we
		# don't train discriminator to produce class labels for generated
		# images (see https://openreview.net/forum?id=rJXTf9Bxg).
		# To preserve sum of sample weights for the auxiliary classifier,
		# we assign sample weight of 2 to the real images.
		disc_sample_weight = [np.ones(2 * 100),
		                      np.concatenate((np.ones(100) * 2,
		                                      np.zeros(100)))]

		epoch_gen_loss = []
		epoch_disc_loss = []

		for index in range(num_batches):
			# 创建用于成成图片的噪声样本
			# 由于batch_size=100，故创建100个100为噪声样本
			noise = np.random.uniform(-1, 1, (100, 100))

			# 真实图片输入
			image_batch = x_train[index * 100:(index + 1) * 100]
			# 真实标签输出
			label_batch = y_train[index * 100:(index + 1) * 100]

			# 随机标签输入
			sampled_labels = np.random.randint(0, 10, 100)

			# 根据噪声和随机标签生成图片
			generated_images = generator.predict(
				[noise, sampled_labels.reshape((-1, 1))], verbose=0)

			# 合并真实图片和生成图片
			x = np.concatenate((image_batch, generated_images))

			# 创建soft真实/生成标签
			soft_zero, soft_one = 0.1, 0.9
			y = np.array([soft_one] * 100 + [soft_zero] * 100)
			# 合并真实图片标签和生成图片标签
			aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

			# 判决器判定
			epoch_disc_loss.append(discriminator.train_on_batch(
				x, [y, aux_y], sample_weight=disc_sample_weight))

			# 生成特征维度为batch_size * 2的噪声
			noise = np.random.uniform(-1, 1, (2 * 100, 100))
			sampled_labels = np.random.randint(0, 10, 2 * 100)

			# we want to train the generator to trick the discriminator
			# For the generator, we want all the {fake, not-fake} labels to say
			# not-fake
			trick = np.ones(2 * 100) * soft_one

			epoch_gen_loss.append(combined.train_on_batch(
				[noise, sampled_labels.reshape((-1, 1))],
				[trick, sampled_labels]))

			progress_bar.update(index + 1)

		print('Testing for epoch {}:'.format(epoch))

		'''evaluate the testing loss here'''

		# generate a new batch of noise
		# test_size
		noise = np.random.uniform(-1, 1, (num_test, 100))

		# sample some labels from p_c and generate images from them
		sampled_labels = np.random.randint(0, 10, num_test)
		generated_images = generator.predict(
			[noise, sampled_labels.reshape((-1, 1))], verbose=False)

		# 真实图片和生成图片
		x = np.concatenate((x_test, generated_images))
		# 真实/生成标签
		y = np.array([1] * num_test + [0] * num_test)
		# 真实标签，成成标签
		aux_y = np.concatenate((y_test, sampled_labels), axis=0)

		# see if the discriminator can figure itself out...
		discriminator_test_loss = discriminator.evaluate(
			x, [y, aux_y], verbose=False)

		discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

		# make new noise
		noise = np.random.uniform(-1, 1, (2 * num_test, 100))
		sampled_labels = np.random.randint(0, 10, 2 * num_test)

		trick = np.ones(2 * num_test)

		generator_test_loss = combined.evaluate(
			[noise, sampled_labels.reshape((-1, 1))],
			[trick, sampled_labels], verbose=False)

		generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

		# generate an epoch report on performance
		train_history['generator'].append(generator_train_loss)
		train_history['discriminator'].append(discriminator_train_loss)

		test_history['generator'].append(generator_test_loss)
		test_history['discriminator'].append(discriminator_test_loss)

		print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
			'component', *discriminator.metrics_names))
		print('-' * 65)

		ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
		print(ROW_FMT.format('generator (train)',
		                     *train_history['generator'][-1]))
		print(ROW_FMT.format('generator (test)',
		                     *test_history['generator'][-1]))
		print(ROW_FMT.format('discriminator (train)',
		                     *train_history['discriminator'][-1]))
		print(ROW_FMT.format('discriminator (test)',
		                     *test_history['discriminator'][-1]))

		# save weights every epoch
		generator.save_weights(
			'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
		discriminator.save_weights(
			'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

		# generate some digits to display
		num_rows = 40
		noise = np.tile(np.random.uniform(-1, 1, (num_rows, 100)),
		                (10, 1))

		sampled_labels = np.array([
			[i] * num_rows for i in range(10)
		]).reshape(-1, 1)

		# get a batch to display
		generated_images = generator.predict(
			[noise, sampled_labels], verbose=0)

		# prepare real images sorted by class label
		real_labels = y_train[(epoch - 1) * num_rows * 10:
		epoch * num_rows * 10]
		indices = np.argsort(real_labels, axis=0)
		real_images = x_train[(epoch - 1) * num_rows * 10:
		epoch * num_rows * 10][indices]

		# display generated images, white separator, real images
		img = np.concatenate(
			(generated_images,
			 np.repeat(np.ones_like(x_train[:1]), num_rows, axis=0),
			 real_images))

		# arrange them into a grid
		img = (np.concatenate([r.reshape(-1, 28)
		                       for r in np.split(img, 2 * 10 + 1)
		                       ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

		Image.fromarray(img).save(
			'plot_epoch_{0:03d}_generated.png'.format(epoch))

	pickle.dump({'train': train_history, 'test': test_history},
	            open('acgan-history.pkl', 'wb'))
