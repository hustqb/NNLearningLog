# coding=utf-8
"""Neural style transfer with Keras.
Run the script with:
```
python neural_style_transfer.py path_to_your_base_image.jpg path_to_your_reference.jpg prefix_for_results
```
e.g.:
```
python neural_style_transfer.py img/tuebingen.jpg img/starry_night.jpg results/my_result
```
Optional parameters:
```
--iter, To specify the number of iterations the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)
```
# Details
Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).
This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":
- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.
- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).
 - The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.
# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
"""

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.engine.topology import get_source_inputs

weightsPath = 'example/hdf5/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

'''1. 命令行读入相关参数'''
parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')  # 内容图地址
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')  # 风格图地址
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')  # 底图地址
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')  # 迭代次数
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')  # 内容权重
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')  # 风格权重
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')  # 底图权重

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
print('base_image_path\'s width={}, height={}'.format(width, height))
print('target_size of ncols={}, nrows={}'.format(img_ncols, img_nrows))

'''2. 预处理与后处理'''


def preprocess_image(image_path):
	"""
	util function to open,
	resize and format pictures into appropriate tensors
	:param image_path:
	:return:
	"""
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	# ：维度扩展，这步在Keras用于图像处理中也很常见，Keras的彩色图输入shape是四阶张量，第一阶是batch_size。
	# 而裸读入的图片是3阶张量。为了匹配，需要通过维度扩展扩充为四阶，第一阶当然就是1了。
	img = np.expand_dims(img, axis=0)
	# vgg提供的预处理，主要完成（1）去均值（2）RGB转BGR（3）维度调换三个任务。
	# 去均值是vgg网络要求的，RGB转BGR是因为这个权重是在caffe上训练的，caffe的彩色维度顺序是BGR。
	# 维度调换是要根据系统设置的维度顺序th/tf将通道维调到正确的位置，如th的通道维应为第二维
	img = vgg19.preprocess_input(img)  # 4
	return img


def deprocess_image(x):
	"""
	util function to convert a tensor into a valid image
	可以看到，后处理的567三个步骤主要就是将#4的预处理反过来了，
	这是为了将处理过后的图片显示出来，resonable。
	:param x:
	:return:
	"""
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, img_nrows, img_ncols))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((img_nrows, img_ncols, 3))
	# Remove zero-center by mean pixel
	x[:, :, 0] += 103.939  # 5
	x[:, :, 1] += 116.779  # 6
	x[:, :, 2] += 123.68  # 7
	# 'BGR'->'RGB'
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')
	return x


'''3. input_tensor = 内容+风格+底图'''
# get tensor representations of our images
# 读入内容和风格图，包装为Keras张量，这是一个常数的四阶张量
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
# 初始化一个待优化图片的占位符，这个地方待会儿实际跑起来的时候要填一张噪声图片进来。
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
# shape=(3, img_nrows, img_ncols, 3)
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

'''4. 加载vgg'''


# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
def load_vgg19(input_):
	# def VGG19(include_top=True, weights='imagenet',
	#           input_tensor=None, input_shape=None,
	#           pooling=None,
	#           classes=1000):
	img_input = Input(tensor=input_, shape=input_.shape)
	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	# Create model.
	inputs = get_source_inputs(input_)
	model = Model(inputs, x, name='vgg19')

	# load weights
	model.load_weights(weightsPath)
	return model


model_vgg = load_vgg19(input_=input_tensor)

print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
# 这是一个张量字典，建立了层名称到层输出张量的映射，
# 通过这个玩意我们可以通过层的名字来获取其输出张量，比较方便。
# 当然不用也行，使用model.get_layer(layer_name).output的效果也是一样的。
outputs_dict = dict([(layer.name, layer.output) for layer in model_vgg.layers])
# 输出：
# {'block4_pool': <tf.Tensor 'block4_pool/MaxPool:0' shape=(3, 25, 25, 512) dtype=float32>,
# ...
# 'block1_pool': <tf.Tensor 'block1_pool/MaxPool:0' shape=(3, 200, 200, 64) dtype=float32>}
print('outputs_dict is {}'.format(outputs_dict))

'''5. 定义损失函数'''
'''compute the neural style loss
	first we need to define 4 util functions'''


def gram_matrix(x):
	"""
	the gram matrix of an image tensor (feature-wise outer product)
	设置Gram矩阵的计算图，
	首先用batch_flatten将输出的featuremap压扁，
	然后自己跟自己做乘法，跟我们之前说过的过程一样。
	注意这里的输入是某一层的representation。
	:param x:
	:return:
	"""
	assert K.ndim(x) == 3
	if K.image_data_format() == 'channels_first':
		# shape=(depth, nrows, ncols)
		features = K.batch_flatten(x)
	else:  # shape=(nrows, ncols, depth)
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram


def style_loss(style, combination):
	"""
	the "style loss" is designed to maintain
	the style of the reference image in the generated image.
	It is based on the gram matrices (which capture style) of
	feature maps from the style reference image
	and from the generated image
	设置风格loss计算方式，以风格图片和待优化的图片的representation为输入。
	计算他们的Gram矩阵，然后计算两个Gram矩阵的差的二范数，除以一个归一化值
	:param style:
	:param combination:
	:return:
	"""
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_nrows * img_ncols
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
	"""
	an auxiliary loss function
	designed to maintain the "content" of the
	base image in the generated image
	设置内容loss计算方式，
	以内容图片和待优化的图片的representation为输入，
	计算他们差的二范数
	:param base:
	:param combination:
	:return:
	"""
	return K.sum(K.square(combination - base))


def total_variation_loss(x):
	"""
	the 3rd loss function, total variation loss,
	designed to keep the generated image locally coherent
	施加全变差正则，全变差正则用于使生成的图片更加平滑自然。
	:param x:
	:return:
	"""
	assert K.ndim(x) == 4
	if K.image_data_format() == 'channels_first':
		a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
		b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
	else:
		a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
		b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))


'''6. 计算loss张量'''
# combine these loss functions into a single scalar
# loss的值是一个浮点数，所以我们初始化一个标量张量来保存它
loss = K.variable(0.)
# layer_features就是图片在模型的block5_conv2这层的输出了，
layer_features = outputs_dict['block5_conv2']
# 计算内容loss取内容图像和待优化图像即可
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

# 与上面的过程类似，只是对多个层的输出作用而已，求出各个层的风格loss，相加即可。
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
	layer_features = outputs_dict[layer_name]
	style_reference_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_reference_features, combination_features)
	loss += (style_weight / len(feature_layers)) * sl

# 求全变差约束，加入总loss中
loss += total_variation_weight * total_variation_loss(combination_image)

'''7. 计算梯度张量'''
# get the gradients of the generated image wrt the loss
# 通过K.grad获取反传梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
# 我们希望同时得到梯度和损失，所以这两个都应该是计算图的输出
if isinstance(grads, (list, tuple)):
	outputs += grads
else:
	outputs.append(grads)

'''8. 编译计算图'''
# 编译计算图。Amazing！我们写了辣么多辣么多代码，
# 其实都在规定输入输出的计算关系，到这里才将计算图编译了。
# 这条语句以后，f_outputs就是一个可用的Keras函数，
# 给定一个输入张量，就能获得其反传梯度了。
f_outputs = K.function([combination_image], outputs)

'''9. 计算loss和grad的标量值'''


# 刚那个优化函数的输出是一个向量
def eval_loss_and_grads(x):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((1, 3, img_nrows, img_ncols))
	else:
		x = x.reshape((1, img_nrows, img_ncols, 3))
	# 激动激动，这里调用了我们刚定义的计算图！
	outs = f_outputs([x])
	# loss就是一个K.various，shape=()
	print('loss_value.shape: ', outs[0].shape)
	# grad是输入张量（底图）在loss函数下的梯度，也就是d_input
	# 所以其shape与输入张量的shape一样，shape=(1, nrows, ncols, 3)
	print('grad_value.shape: ', outs[1].shape)
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')
	return loss_value, grad_values


# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):
	def __init__(self):
		# 这个类别的事不干，专门保存损失值和梯度值
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		# 调用刚才写的那个函数同时得到梯度值和损失值，但只返回损失值，
		# 而将梯度值保存在成员变量self.grads_values中，这样这个函数就满足了func要求的条件
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		# 这个函数不用做任何计算，只需要把成员变量self.grads_values的值返回去就行了
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


'''10. 迭代优化，优化方法L-BFGS-B'''
evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
# 根据后端初始化一张噪声图片，做去均值
x_main = preprocess_image(base_image_path)

for i in range(iterations):
	print('Start of iteration', i)
	start_time = time.time()
	# 更新x
	# evaluator.loss和evaluator.grad是返回标量值的callable对象
	# x_main.flatten()是一个1D的ndarray
	x_main, min_val, info = fmin_l_bfgs_b(evaluator.loss, x_main.flatten(),
	                                      fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	# save current generated image
	# 每次迭代完成后把输出的图片后处理一下，保存起来
	img_main = deprocess_image(x_main.copy())
	fname = result_prefix + '_at_iteration_%d.png' % i
	imsave(fname, img_main)
	end_time = time.time()
	print('Image saved as', fname)
	print('Iteration %d completed in %ds' % (i, end_time - start_time))
