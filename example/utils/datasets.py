# coding=utf-8
"""
获取数据集
	cifar10  10个类别的图片
"""
import os
import numpy as np
from six.moves import cPickle


def load_mnist():
	path = os.path.join(os.getcwd(), 'famousData/mnist.npz')
	data = np.load(path)
	return (data['x_train'], data['y_train']), \
	       (data['x_test'], data['y_test'])


def load_batch(fpath, label_key='labels'):
	"""Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
	f = open(fpath, 'rb')
	d = cPickle.load(f)
	f.close()
	data = d['data']
	labels = d[label_key]
	data = data.reshape(data.shape[0], 32, 32, 3)
	return data, labels


def load_cifar10():
	cifar10_path = r'/home/zqb/data/cifar-10-batches-py'
	num_train_samples = 50000
	x_train = np.zeros((num_train_samples, 32, 32, 3), dtype='uint8')
	y_train = np.zeros((num_train_samples,), dtype='uint8')
	
	for i in range(1, 6):
		fpath = os.path.join(cifar10_path, 'data_batch_' + str(i))
		data, labels = load_batch(fpath)
		x_train[(i - 1) * 10000:i * 10000, :, :, :] = data
		y_train[(i - 1) * 10000:i * 10000] = labels
	
	fpath = os.path.join(cifar10_path, 'test_batch')
	x_test, y_test = load_batch(fpath)
	y_train = np.reshape(y_train, (len(y_train), 1))
	y_test = np.reshape(y_test, (len(y_test), 1))
	return (x_train, y_train), (x_test, y_test)
		
		
if __name__ == '__main__':
	temp_data, temp_labels = load_batch(r'/home/zqb/data/cifar-10-batches-py/data_batch_1')
	print(temp_data)
	print(temp_labels)
	print(temp_data.shape)
	print(len(temp_labels))
