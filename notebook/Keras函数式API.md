# Keras的函数式模型
参考[Keras中文文档](http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/)
相较于贯序模型，Keras的函数式模型更灵活，可以完全取代贯序模型
## 简介
Keras函数式模型接口是用户定义多输出模型、非循环有向模型
或具有共享层的模型等复杂模型的途径。
一句话，只要你的模型不是类似VGG一样一路走到黑的模型，
或者你的模型需要多于一个的输出，那么你总应该选择函数式模型。
函数式模型时最广泛的一类模型，贯序模型（Sequential）只是它的一种特殊情况。

## 第一个模型：全连接网络
几个概念：
- 层对象接受张量为参数，返回一个张量。
- 输入是张量，输出也是张量的一个框架就是一个模型，通过Model定义
- 这样的模型可以被像Keras的Sequential一样被训练
```python
from keras.layers import Input, Dense
from keras.models import Model

data = ''
labels = ''

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels)
```
## 所有的模型都是可调用的，就像层一样
利用函数式模型的接口，我们可以很容易的重用已经训练好的模型：
你可以把模型当做一个层样，通过提供一个tensor来调用。
注意当你调用一个模型，你不仅仅重用了它的结构，
也**重用了它的权重**。
```python
x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)
```
这种方式允许你快速的创建能处理序列信号的模型，
你可以很快讲一个图像分类模型变为一个对视频分类的模型，
只需要一行代码：
```python
from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```
## 多输入和多输出模型
使用函数式模型的一个典型场景是搭建多输入、多输出的模型。

考虑这样一个模型。我们希望预测Twitter上一条新闻会被转发和点赞多少次。
模型的主要输入是新闻本身，也就是一个词语的序列。
但我们还可以拥有额外的输入，如新闻发布的日期等。

这个模型的损失函数将由两部分组成，
辅助的损失函数评估仅仅基于新闻本身做出预测的情况，
主损失函数评估基于新闻和额外信息的预测的情况，
即使来自主损失函数的梯度发生弥散，
来自辅助损失函数的信息也能够训练Embeddding和LSTM层。
在模型中早点使用主要的损失函数是对于深度网络的一个良好的正则方法。
总而言之，该模型框图如下：
![模型框图](https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png)
让我们用函数式模型来实现这个框图

主要的输入接受新闻本身，即一个整数的序列（每个整数编码了一个词）。
这些整数位于1到10000之间（即我们的字典有10000个词）。
这个序列有100个单词。
```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

main_input = Input(shape=(100,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
```
然后，我们插入一个额外的损失，使得即使在主损失很高的情况下，
LSTM和Embedding层也可以平滑的训练。
```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```
再然后，我们将LSTM与额外的输入数据串联起来组成输入，送入模型中：
```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```
最后，我们定义整个2输入，2输出的模型：
```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```
模型定义完毕，下一步编译模型。我们给额外的损失赋0.2的权重。
我们可以通过关键字参数`loss_weight`或`loss`来为不同的输出设置不同的损失函数和权值。
这两个参数均可为Python的列表或字典。
这里我们给`loss`传递单个损失函数，
这个损失函数或被应用于所有输出上。
```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weight=[1., 0.2])
```
编译完成后，我们通过传递训练数据和目标值训练该模型：
```python
model.fit([headline_data, additional_data], [labels, labels], epochs=50, batch_size=32)
```
因为我们输入和输出是被命名过的，我们也可以用下面的方式编译和训练模型：
```python
model.compile(optimizer='rmsprop', 
loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'}, 
loss_weight={'main_output': 1., 'aux_output': 0.2})

model.fit({'main_input': headline_data, 'aux_input': additional_data}, 
{'main_output': labels, 'aux_output': labels}, 
epochs=50, batch_size=32)
```
