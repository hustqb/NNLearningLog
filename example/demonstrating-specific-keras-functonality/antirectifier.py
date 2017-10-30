# coding=utf-8
"""展示如何自定义中间层。
自定义一个中间层——Antirectifier，该层会修改tensor shape。
因此，需要指定两个函数：compute_output_shape和call。

PS：Lambda层也能实现相同的功能。

关键：使用Keras后端库——backend
"""
from __future__ import print_function
# from utils.datasets import load_mnist
from example.utils.datasets import load_mnist
import keras
from keras.models import Model
from keras import layers
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 40


class Antirectifier(layers.Layer):
	"""对于输入数据的正值部分和负值部分，取L2范数，同时，
	输出tensor shape是输入的两倍，目的是为了取代ReLU。
	2D tensor of shape (samples, n)
	2D tensor of shape (samples, 2*n)

	理论依据：
	假设前一层输出分布的均值近似为0，在应用ReLU激活函数时，
	我们会舍弃一半的数据（小于0的部分），这种做法比较低效。

	Antirectifier可以像ReLU一样返回值都大于0，且不用舍弃数据。

	用MNIST数据集来测试可以看到，相较于基于ReLU的网络，
	基于Antirectifier可以用更少的神经元数目得到一个相当的accuracy。
	"""

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(shape) == 2  # 只对2维张量有效
		shape[-1] *= 2
		return tuple(shape)

	def call(self, inputs, **kwargs):
		inputs -= K.mean(inputs, axis=1, keepdims=True)
		inputs = K.l2_normalize(inputs, axis=1)
		pos = K.relu(inputs)
		neg = K.relu(-inputs)
		return K.concatenate([pos, neg], axis=-1)


(x_train, y_train), (x_test, y_test) = load_mnist()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_data = layers.Input((784,))
x = layers.Dense(256)(input_data)
x = Antirectifier()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(256)(x)
x = Antirectifier()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(num_classes)(x)
x = layers.Activation('softmax')(x)
model = Model(input_data, x)

model.compile('rmsprop', 'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
