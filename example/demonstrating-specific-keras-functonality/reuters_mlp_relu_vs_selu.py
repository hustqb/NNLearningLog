# coding=utf-8
"""
对于路透视新闻主题分类，训练一个简单的MLP网络，
然后使用并比较两种不同的激活函数：ReLU和SELU。
"""
from __future__ import print_function
import sys
sys.path.append('/home/zqb/zqb_code/NNLearningLog/example')
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras
from utils import datasets
from keras.models import Model
from keras.layers import Dense, Input, Activation, Dropout
from keras.layers.noise import AlphaDropout
from keras.preprocessing.text import Tokenizer

maxWords = 1000
batch_size = 16
epochs = 40
plot = True


def create_network(n_dense=6, dense_units=16,
                   activation='selu', dropout=AlphaDropout,
                   dropout_rate=0.1, kernel_initializer='lecun_normal',
                   optimizer='adam', num_classes=1, max_words=maxWords):
	"""Generic function to create a fully-connected neutral network"""
	input_data = Input((max_words,))
	x = Dense(dense_units, kernel_initializer=kernel_initializer)(input_data)
	x = Activation(activation)(x)
	x = dropout(dropout_rate)(x)

	for i in range(n_dense - 1):
		x = Dense(dense_units, kernel_initializer=kernel_initializer)(x)
		x = Activation(activation)(x)
		x = dropout(dropout_rate)(x)
	x = Dense(num_classes, activation='softmax')(x)
	model = Model(input_data, x)
	model.compile(optimizer, 'categorical_crossentropy',
	              metrics=['accuracy'])
	return model


network1 = {
	'n_dense': 6,
	'dense_units': 16,
	'activation': 'relu',
	'dropout': Dropout,
	'dropout_rate': 0.5,
	'kernel_initializer': 'glorot_uniform',
	'optimizer': 'sgd'
}

network2 = {
	'n_dense': 6,
	'dense_units': 16,
	'activaton': 'selu',
	'dropout': AlphaDropout,
	'dropout_rate': 0.1,
	'kernel_initializer': 'lecun_normal',
	'optimizer': 'sgd'
}

(x_train, y_train), (x_test, y_test) = datasets.load_reuter(num_words=maxWords,
                                                            test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=maxWords)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('\nBuilding network 1...')

model1 = create_network(num_classes=num_classes, **network1)
history_model1 = model1.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)

score_model1 = model1.evaluate(x_test,
                               y_test,
                               batch_size=batch_size,
                               verbose=1)

print('\nBuilding network 2...')
model2 = create_network(num_classes=num_classes, **network2)

history_model2 = model2.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)

score_model2 = model2.evaluate(x_test,
                               y_test,
                               batch_size=batch_size,
                               verbose=1)

print('\nNetwork 1 results')
print('Hyperparameters:', network1)
print('Test score:', score_model1[0])
print('Test accuracy:', score_model1[1])
print('Network 2 results')
print('Hyperparameters:', network2)
print('Test score:', score_model2[0])
print('Test accuracy:', score_model2[1])

plt.plot(range(epochs),
         history_model1.history['val_loss'],
         'g-',
         label='Network 1 Val Loss')
plt.plot(range(epochs),
         history_model2.history['val_loss'],
         'r-',
         label='Network 2 Val Loss')
plt.plot(range(epochs),
         history_model1.history['loss'],
         'g--',
         label='Network 1 Loss')
plt.plot(range(epochs),
         history_model2.history['loss'],
         'r--',
         label='Network 2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('comparison_of_networks.png')
