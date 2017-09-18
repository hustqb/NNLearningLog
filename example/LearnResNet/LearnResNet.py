# coding=utf-8
from __future__ import print_function

"""
follow 菜鸡的啄米日常/使用keras搭建残差网络
"""
import numpy as np
import json
import keras.backend as K
from keras.layers import add, Input, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, _obtain_input_shape

'''BatchNormalization是在normalization这个模块里，这个模块目前就这一个层。
我们导入的是函数merge而不是层Merge，在构造函数式模型时要使用函数merge，
因为函数式模型的整个构造过程就是对张量的操作。
最后注意Input不是一个层，而是一个函数，函数的返回值是一个张量。'''

weights_path = 'resnet50_weights.h5'  # 预训练的权重
weights_notop_path = 'resnet50_weights_notop.h5'
fpath = 'imagenet_class_index.json'  # 分类编号对照表
img_path = 'elephant.jpg'  # 测试图片（这里可以改成任何你想测试的图片)
bn_axis = 3


def identity_block(input_tensor, kernel_size, filters, stage, block):
	"""
	The identity block is the block that has no conv layer at shortcut.
	:param input_tensor:
	:param kernel_size: defualt 3
	:param filters: list of integers, the filterss of 3 conv layer at main path
	:param stage: integer, current stage label, used for generating layer names
	:param block: 'a', 'b', ..., current block label, used for generating layer names
	:return: Output tensor for the block
	"""
	'''padding'''
	filters1, filters2, filters3 = filters  # 三个卷积层核的数目
	'''style + stage + block + _branch'''
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	'''caffe中的BN层分两步完成，所以结构图里除了BN还有一个scale，
	Keras是BN就是BN。
	该层在每个batch上将前一层的激活值重新规范化，
	mean=0, std=1'''
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)
	
	x = Conv2D(filters2, kernel_size, padding='same',
	           name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)
	
	x = Conv2D(filters3, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
	
	'''A list of input tensors (at least 2).'''
	x = add([x, input_tensor])
	x = Activation('relu')(x)
	return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
	"""
	conv_block is the block that has a conv layer at shortcut.
	:param input_tensor:
	:param kernel_size: defualt 3
	:param filters: list of integers, the filterss of 3 conv layer at main path
	:param stage: integer, current stage label, used for generating layer names
	:param block: 'a', 'b', ..., current block label, used for generating layer names
	:param strides: An integer or tuple/list of a single integer
	:return:
	"""
	filters1, filters2, filters3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	x = Conv2D(filters1, 1, strides=strides,
	           name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)
	
	x = Conv2D(filters2, kernel_size, padding='same',
	           name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)
	
	x = Conv2D(filters3, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
	
	shortcut = Conv2D(filters3, 1, strides=strides,
	                  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis,
	                              name=bn_name_base + '1')(shortcut)
	
	x = add([x, shortcut])
	x = Activation('relu')(x)
	return x


def res_net50(include_top=True, weights='imagenet', input_tensor=None,
              input_shape=None, pooling=None, classes=1000):
	"""
	Instantiates the ResNet50 architecture.
	:param include_top: whether to include the fully-connected layer at the top of the network.
	:param weights:one of 'None' (random initialization) or "imagenet" (pre-training)
	:param input_tensor:
	:param input_shape:
	:param pooling: Optional pooling mode for feature extracion when 'include_top' is 'False'.
				    'None' means that the output of the model will be the 4D tensor output
				    of the last convolutional layer.
				    'avg' means that golobal average pooling will be applied to the output
				    of the last convolutional layer, and thus the output of the model will
				    be a 2D tensor.
				    'max' means that global max pooling will be applied.
	:param classes: optional number of classes to classify images into, only to be specified
					if 'include_top' is True, and if no 'weights' argument is specified.
	:return: A Keras moder instance.
	"""
	if weights not in {'imagenet', 'None'}:
		raise ValueError('The weight argument should be eighter None(random initialization) '
		                 'or imagenet(pre-training on ImageNet).')
	if weights == 'imagenet' and include_top and classes != 1000:
		raise ValueError('If usinig weights as imagenet with include_top as true, '
		                 'classes should be 1000')
	
	'''Detemine proper input shape
	以tensorflow为后端'''
	input_shape = _obtain_input_shape(input_shape,
	                                  default_size=224,
	                                  min_size=197,
	                                  data_format=K.image_data_format(),
	                                  include_top=include_top)
	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor
	global bn_axis
	if K.image_data_format() != 'channels_last':
		bn_axis = 1
	
	x = Conv2D(64, (7, 7), strides=2, padding='same', name='conv1')(img_input)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
	x = AveragePooling2D((7, 7), name='avg_pool')(x)
	
	if include_top:
		x = Flatten()(x)
		x = Dense(classes, activation='softmax', name='fc1000')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling2D()(x)
		elif pooling == 'max':
			x = GlobalMaxPool2D()(x)
	model = Model(img_input, x, name='resnet50')
	
	'''load weights'''
	if weights == 'imagenet':
		model.load_weights(weights_path)
		if K.backend() == 'theano':
			raise ValueError('Not support theano yet.'
			                 'Please turn to tensorflow.')
		if K.image_data_format() == 'channel_first':
			raise ValueError('You are using the TensorFlow backend, '
			                 'yet you are using the Theano '
			                 'image data format convention'
			                 '(image_data_format="channel_first").')
	return model


def my_decode(preds, top=5):
	CLASS_INDEX = json.load(open(fpath))
	results = []
	for pred in preds:
		top_indices = pred.argsort()[-top:][::-1]
		result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
		result.sort(key=lambda x: x[2], reverse=True)
		results.append(result)
	return results


if __name__ == '__main__':
	'''
	1. 使用少量样本训练一个resnet网络并判断样本
		1.1 电量样本（n_samples, 28, 3, 1)
	2. 调整标签后，使用之前的权重经少量训练后再次判断样本
	3. 使用大量样本，使用之前的权重
	4. 调整标签
	5. 使用大量样本，使用之前的权重——finished'''
	bk_data_shape = (1, 1, 1, 1)  # 数据格式(样本数，axis0，axis1，axis2)
	bk_classes = 1000  # 故障的种类
	model_main = res_net50(include_top=True,  # 是否包含fc层
	                       weights='imagenet',  # 是否使用预训练的权重
	                       input_tensor=None,  # 有无输入张量
	                       input_shape=None,   # 输入张量的shape
	                       pooling=None,   # 池化方式：avg/max
	                       classes=1000)  # 输出类别数目
	
	img = image.load_img(img_path, target_size=(224, 224))
	x_main = image.img_to_array(img)
	x_main = np.expand_dims(x_main, axis=0)
	x_main = preprocess_input(x_main)
	print('Input image shape:', x_main.shape)
	
	preds_main = model_main.predict(x_main)
	print('Predicted:', my_decode(preds_main))
