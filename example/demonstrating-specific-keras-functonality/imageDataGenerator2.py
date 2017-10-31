# coding=utf-8
"""
keras.preprocessing.image.ImageDataGenerator
图像生成器，按批次实时的生成训练数据
从本地文件系统的包含各种图片的文件夹中读取数据，创建生成器。
由于下面代码中路径："data/train"和"data/validation"没有创建，
所以代码不能运行。
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

train_datagen = ImageDataGenerator(
	rescale=1./255,  # If None or 0, no rescaling is applied, otherwise we
	# multiply the data by the value provided(before applying any other transformation).
	shear_range=0.2,  # Shear Intensity(Shear angle in counter-clockwise direction
	# as radians).逆时针切去0.2弧度
	zoom_range=0.2,  # Range for random zoom. If a float,
	# [lower, upper] = [1-zoom_range, 1+zoom_range].
	horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory(directory): Takes the path to a directory, and generates
# batches of augmented/normalized data. Yield batches indefinitely,
# in an infinite loop.
train_generator = train_datagen.flow_from_directory(
	'data/train',  # path to the target directory.
	target_size=(150, 150),  # (height, width)
	batch_size=32,
	class_mode='binary'  # one of "categorical", "binary", "sparse" or None.
)
validation_generator = test_datagen.flow_from_directory(
	'data/validation',
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary'
)

# get cnn model
model = cnn((32, 32, 3))
model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
	train_generator,  # A generator
	steps_per_epoch=2000,  # Total number of steps(batches of samples) to
	# yield from generator before declaring one epoch finished and starting
	# the next epoch. It should typically be equal to the number of
	# unique samples of your dataset divided by the batch size.
	epochs=50,
	validation_data=validation_generator,
	validation_steps=800
)
