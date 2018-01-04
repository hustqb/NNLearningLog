# coding=utf-8
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DCGAN(object):
	def __init__(self):

		self.D = None  # discriminator
		self.G = None  # generator
		self.AM = None  # adversarial model
		self.DM = None  # discriminator model

	# (W−F+2P)/S+1
	def discriminator(self):
		"""判别器"""
		if self.D:
			return self.D
		self.D = Sequential()
		# In: 28 x 28 x 1, depth = 1
		# Out: 14 x 14 x 1, depth=64
		self.D.add(Conv2D(64, 5, strides=2,
		                  input_shape=(28, 28, 1),
		                  padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(0.4))

		self.D.add(Conv2D(128, 5, strides=2,
		                  padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(0.4))

		self.D.add(Conv2D(256, 5, strides=2,
		                  padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(0.4))

		self.D.add(Conv2D(512, 5, strides=1,
		                  padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(0.4))

		# Out: 1-dim probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))
		self.D.summary()
		return self.D

	def generator(self):
		if self.G:
			return self.G
		self.G = Sequential()

		# In: 100
		# Out: dim x dim x depth
		self.G.add(Dense(7 * 7 * 256, input_dim=100))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))
		self.G.add(Reshape((7, 7, 256)))
		self.G.add(Dropout(0.4))

		# In: dim x dim x depth
		# Out: 2*dim x 2*dim x depth/2
		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(128, 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(64, 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(Conv2DTranspose(32, 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
		self.G.add(Conv2DTranspose(1, 5, padding='same'))
		self.G.add(Activation('sigmoid'))
		self.G.summary()
		return self.G

	def discriminator_model(self):
		if self.DM:
			return self.DM
		optimizer = RMSprop(lr=0.0002, decay=6e-8)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def adversarial_model(self):
		if self.AM:
			return self.AM
		optimizer = RMSprop(lr=0.0001, decay=3e-8)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.AM


class MNIST_DCGAN(object):
	def __init__(self):
		self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
		self.x_train = self.x_train.reshape(-1, 28, 28, 1).astype(np.float32)

		self.DCGAN = DCGAN()
		self.discriminator = self.DCGAN.discriminator_model()
		self.adversarial = self.DCGAN.adversarial_model()
		self.generator = self.DCGAN.generator()

	def train(self, train_steps=2000, batch_size=256, save_interval=0):
		noise_input = None
		losses = []
		if save_interval > 0:
			noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
		for i in range(train_steps):
			images_train = self.x_train[np.random.randint(0,
			                                              self.x_train.shape[0],
			                                              size=batch_size), :, :, :]
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			images_fake = self.generator.predict(noise)
			x = np.concatenate((images_train, images_fake))
			y = np.ones([2 * batch_size, 1])
			y[batch_size:, :] = 0
			d_loss = self.discriminator.train_on_batch(x, y)

			y = np.ones([batch_size, 1])
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			a_loss = self.adversarial.train_on_batch(noise, y)
			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
			print(log_mesg)
			losses.append(d_loss + a_loss)
			if save_interval > 0:
				if (i + 1) % save_interval == 0:
					self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))
		losses = pd.DataFrame(losses, columns=['d_loss', 'd_acc', 'a_loss', 'a_acc'])
		figname = 'losses.png'
		fig = plt.figure(figsize=(12, 8))
		ax = plt.subplot()
		losses.plot(kind='line', ax=ax, alpha=0.7)
		plt.savefig(figname)
		plt.close()

	def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
		filename = 'mnist.png'
		if fake:
			filename = 'fake_mnist.png'
			if noise is None:
				noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
			else:
				filename = "mnist_%d.png" % step
			images = self.generator.predict(noise)
		else:
			i = np.random.randint(0, self.x_train.shape[0], samples)
			images = self.x_train[i, :, :, :]

		plt.figure(figsize=(10, 10))
		for i in range(images.shape[0]):
			plt.subplot(4, 4, i + 1)
			image = images[i, :, :, :]
			image = np.reshape(image, [28, 28])
			plt.imshow(image, cmap='gray')
			plt.axis('off')
		plt.tight_layout()
		if save2file:
			plt.savefig(filename)
			plt.close('all')
		else:
			plt.show()


if __name__ == '__main__':
	mnist_dcgan = MNIST_DCGAN()
	mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
	mnist_dcgan.plot_images(fake=True, save2file=True)
	mnist_dcgan.plot_images(fake=False, save2file=True)
