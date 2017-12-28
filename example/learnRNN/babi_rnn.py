# coding=utf-8
"""
Trains two recurrent neural networks based upon a story and a question.
The resulting merged vector is then queried to answer a range of bAbI tasks.
The results are comparable to those for an LSTM model provided in Weston et al.:
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
这里有很多数据，该程序只用了一份，准确率：0.264
"""

from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
	"""Return the tokens of a sentence including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	"""
	# \W匹配非字母数字和下划线
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
	"""
	Parse stories provided in the bAbi tasks format
	If only_supporting is true,
	only the sentences that support the answer are kept.
	"""
	data = []
	story = []
	for line in lines:
		line = line.decode('utf-8').strip()
		nid, line = line.split(' ', 1)
		nid = int(nid)
		if nid == 1:
			story = []
		if '\t' in line:
			q, a, supporting = line.split('\t')
			q = tokenize(q)
			substory = None
			if only_supporting:
				# Only select the related substory
				supporting = map(int, supporting.split())
				substory = [story[i - 1] for i in supporting]
			else:
				# Provide all the substories
				substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append('')
		else:
			sent = tokenize(line)
			story.append(sent)
	return data


def get_stories(f, only_supporting=False, max_length=None):
	"""
	Given a file name, read the file, retrieve the stories,
	and then convert the sentences into a single story.
	If max_length is supplied,
	any stories longer than max_length tokens will be discarded.
	"""
	data = parse_stories(f.readlines(), only_supporting=only_supporting)
	flatten = lambda data: reduce(lambda x, y: x + y, data)
	data = [(flatten(story), q, answer) for story, q, answer in data if
	        not max_length or len(flatten(story)) < max_length]
	return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
	xs = []
	xqs = []
	ys = []
	for story, query, answer in data:
		x = [word_idx[w] for w in story]
		xq = [word_idx[w] for w in query]
		# let's not forget that index 0 is reserved
		y = np.zeros(len(word_idx) + 1)
		y[word_idx[answer]] = 1
		xs.append(x)
		xqs.append(xq)
		ys.append(y)
	return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xq, maxlen=query_maxlen), np.array(ys)


RNN = recurrent.LSTM
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           50,
                                                           100,
                                                           100))

path = get_file('babi-tasks-v1-2.tar.gz',
                origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)
# QA2 with 1000 samples
challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
train = get_stories(tar.extractfile(challenge.format('train')))
test = get_stories(tar.extractfile(challenge.format('test')))
'''
In [21]: print(train[0][0])
[u'Mary', u'moved', u'to', u'the', u'bathroom', u'.', 
u'Sandra', u'journeyed', u'to', u'the', u'bedroom', u'.', 
u'Mary', u'got', u'the', u'football', u'there', u'.', 
u'John', u'went', u'to', u'the', u'kitchen', u'.', 
u'Mary', u'went', u'back', u'to', u'the', u'kitchen', u'.', 
u'Mary', u'went', u'back', u'to', u'the', u'garden', u'.']

In [22]: print(train[0][1])
[u'Where', u'is', u'the', u'football', u'?']

In [23]: print(train[0][2])
garden
'''

vocab = set()
for story, q, answer in train + test:
	vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('vocab = {}'.format(vocab))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')

sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, 50)(sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

question = layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, 50)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
encoded_question = RNN(50)(encoded_question)
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

merged = layers.add([encoded_sentence, encoded_question])
merged = RNN(50)(merged)
merged = layers.Dropout(0.3)(merged)
preds = layers.Dense(vocab_size, activation='softmax')(merged)

model = Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([x, xq], y,
          batch_size=32,
          epochs=40,
          validation_split=0.05)
loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=32)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
