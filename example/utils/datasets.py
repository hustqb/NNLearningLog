# coding=utf-8
"""
获取数据集
	cifar10  10个类别的图片
"""
import os
import json
import numpy as np
from six.moves import cPickle


def load_mnist():
	path = os.path.join(os.getcwd(), 'famousData/mnist.npz')
	data = np.load(path)
	return (data['x_train'], data['y_train']), \
	       (data['x_test'], data['y_test'])


def load_imdb(num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3):
	"""Loads the IMDB dataset.
	# Arguments
		num_words: max number of words to include. Words are ranked
			by how often they occur (in the training set) and only
			the most frequent words are kept
		skip_top: skip the top N most frequently occurring words
			(which may not be informative).
		maxlen: truncate sequences after this length.
		seed: random seed for sample shuffling.
		start_char: The start of a sequence will be marked with this character.
			Set to 1 because 0 is usually the padding character.
		oov_char: words that were cut out because of the `num_words`
			or `skip_top` limit will be replaced with this character.
		index_from: index actual words with this index and higher.
	# Returns
		Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
	# Raises
		ValueError: in case `maxlen` is so low
			that no input sequence could be kept.
	Note that the 'out of vocabulary' character is only used for
	words that were present in the training set but are not included
	because they're not making the `num_words` cut here.
	Words that were not seen in the training set but are in the test set
	have simply been skipped.
	"""
	path = os.path.join(os.getcwd(), 'famousData/imdb.npz')
	with np.load(path) as f:
		x_train, labels_train = f['x_train'], f['y_train']
		x_test, labels_test = f['x_test'], f['y_test']

	np.random.seed(seed)
	np.random.shuffle(x_train)
	np.random.seed(seed)
	np.random.shuffle(labels_train)

	np.random.seed(seed * 2)
	np.random.shuffle(x_test)
	np.random.seed(seed * 2)
	np.random.shuffle(labels_test)

	xs = np.concatenate([x_train, x_test])
	labels = np.concatenate([labels_train, labels_test])

	if start_char is not None:
		xs = [[start_char] + [w + index_from for w in x] for x in xs]
	elif index_from:
		xs = [[w + index_from for w in x] for x in xs]

	# temp_xs = []
	# if start_char is not None:
	# 	temp_xs = [start_char]
	# for x in xs:  # x为样本中的序列
	# 	for w in x:  # w为序列中的单词
	# 		temp_xs += [w + index_from]
	# xs = temp_xs

	if maxlen:
		xs, labels = _remove_long_seq(maxlen, xs, labels)
		if not xs:
			raise ValueError('After filtering for sequences shorter than maxlen=' +
			                 str(maxlen) + ', no sequence was kept. '
			                               'Increase maxlen.')
	if not num_words:
		num_words = max([max(x) for x in xs])

	# by convention, use 2 as OOV word
	# reserve 'index_from' (=3 by default) characters:
	# 0 (padding), 1 (start), 2 (OOV)
	if oov_char is not None:
		xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
	else:
		xs = [[w for w in x if (skip_top <= w < num_words)] for x in xs]

	idx = len(x_train)
	x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
	x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

	return (x_train, y_train), (x_test, y_test)


def _remove_long_seq(maxlen, seq, label):
	new_seq, new_label = [], []
	for x, y in zip(seq, label):
		if len(x) < maxlen:
			new_seq.append(x)
			new_label.append(y)
	return new_seq, new_label


def get_word_index():
	path = os.path.join(os.getcwd(), 'famousData/imdb_word_index.json')
	f = open(path)
	data = json.load(f)
	f.close()
	return data


def load_reuter(num_words=None, skip_top=0, maxlen=None, test_split=0.2,
                seed=113, start_char=1, oov_char=2, index_from=3):
	"""Loads the Reuters newswire classification dataset."""
	with np.load('/home/zqb/zqb_code/NNLearningLog/famousData/reuters.npz') as f:
		xs, labels = f['x'], f['y']

	np.random.seed(seed)
	np.random.shuffle(xs)
	np.random.seed(seed)
	np.random.shuffle(labels)
	if start_char is not None:
		xs = [[start_char] + [w + index_from for w in x] for x in xs]
	elif index_from:
		xs = [[w + index_from for w in x] for x in xs]
	if not num_words:
		num_words = max([max(x) for x in xs])
	# by convention, use 2 as OOV word
	# reserve 'index_from' (=3 by default) characters:
	# 0(padding), 1(start), 2(OOV)
	if oov_char is not None:
		xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
	else:
		xs = [[w for w in x if (skip_top <= w < num_words)] for x in xs]
	idx = int(len(xs) * (1 - test_split))
	x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
	x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

	return (x_train, y_train), (x_test, y_test)


def get_reuter_index():
	"""Retrieves the dictionary mapping word indices back to words."""
	f = open('/home/zqb/zqb_code/NNLearningLog/famousData/reuters_word_index.json')
	data = json.load(f)
	f.close()
	return data


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
	cifar10_path = r'/home/zqb/zqb_code/NNLearningLog/famousData/cifar-10-batches-py'
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
	temp_data, temp_labels = load_batch(r'/home/zqb/zqb_code/NNLearningLog/famousData/cifar-10-batches-py/data_batch_1')
	print(temp_data)
	print(temp_labels)
	print(temp_data.shape)
	print(len(temp_labels))
