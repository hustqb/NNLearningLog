##Keras学习笔记
摘自[知乎-啄米日常](https://zhuanlan.zhihu.com/chicken-life)
###基本知识
####符号式编程
符号式编程意思就是，先确定符号以及符号之间的计算关系，然后再放数据进去计算的办法。
<font color=red>符号之间的运算关系，就称为运算图。</font>
它的一大优点是，当确立了输入和输出的计算关系后，在进行运算前我们可以对这种运算关系进行自动化简，
从而减少计算量，提高计算速度。另一个优势是，
运算图一旦确定，整个计算过程就都清楚了，可以ongoing内存复用的方式减少程序占用的内存。
#### 张量
在Keras，Theano和TensorFlow中，参与符号运算的那些变量统一称为张量。
张量是矩阵的推广。
<font color=red>Keras的计算过程就是建立一个从张量到张量的映射函数，然后再放入真实数据进行计算</font>。
对深度学习而言，这个“映射函数”就是一个神经网络，而神经网络中的每个层自然也都是从张量到张量的映射。
### Keras框架结构
backend + models + layers
#### Keras创建并训练神经网络
注：只针对函数式编
1. 创建layer（定义层的输入张量到输出张量的映射）
2. 创建model（定义model的输入张量到输出张量的映射）
3. model.compile(loss=?, optimizers=?)
4. model.fit(x_train, y_train)

此外，keras提供了一组模块来对神经网络进行配置：
- initialization
- regularizers
- constraints

为了方便调试、分析和使用网络，处理数据，Keras提供了下面的模块：
- callbacks
- visualization
- preprocessing
- utils

最后，为了能让用户上手就能跑一些模型，Keras提供了一个常用数据库的模块：
- datasets

PS: 如果用户希望将Keras与scikit-learn联动，Keras也提供了这种联动机制：
- wrappers.scikit-learn

### backend
可能会用到的函数：
- fuction: 用于讲一个计算图（计算关系）编译为具体的函数。
典型的应用场景是输出网络的中间层结果。
- image_ordering和set_image_ordering: 这组函数用于返回、设置图片的维度顺序，
由于Theano和Tensorflow的图片维度顺序不一样，所以有时候需要获取/指定。典型应用是
当希望网络自适应的根据使用的后端调整图片维度顺序时。
- learning_phase: 这个函数的主要作用是返回网络的运行状态，0代表测试，1代表训练。
当你需要便携一个在训练和测试是行为不同的层（如Dropout）时，它会很有用。
- gradients: 求损失函数关于变量的导数，也就是网络的反向计算过程。
这个函数在不训练网络而只想用梯度做一点奇怪的事情的时候会很有用，如图像风格转移。
### models/layers
模型有两套训练和测试的函数，一套是fit，evaluate等，
另一套是fit_generator，evaluate_generator，前者适用于普通情况，后者适用于数据是以迭代器动态生成的情况。
**对模型而言，最核心的函数**有两个：
- compile(): 编译，模型在训练前必须编译，
这个函数用于完成添加正则项啊，确定目标函数啊，确定优化器啊等等一系列模型配置功能。
这个函数必须指定的参数是优化器和目标函数，经常还需要指定一个metrics来评价模型。
- fit()/fit_generator(): 用来训练模型，参数较多，是需要重点掌握的函数。

另外，**模型还有几个常用的属性和函数**：
- layers: 
- get_layer(): 
- pop(): 这个函数文档里没有，但是可以用。作用是弹出模型的最后一层，
从前进行finetune时没有pop，大家一般用model.layers.pop()来完成同样的功能。

Keras的**层对象**是构筑模型的基石，
除了卷积层，递归神经网络层，全连接层，激活层这种烂大街的Layer对象外，
Keras还有一些不是那么烂大街的东西：
- Advanced Activation：高级激活层，主要收录了包括leakyReLU，pReLU，ELU，SReLU等一系列高级激活函数，
这些激活函数不是简单的element-wise计算，所以单独拿出来实现一下
- Merge层：这个层用于将多个层对象的输出组合起来，支持级联、乘法、余弦等多种计算方式，
它还有个小兄弟叫merge，这个函数完成与Merge相同的作用，但输入的对象是张量而不是层对象。
不过在新版Keras中已被弃用，改为直接使用Add, Concatenate.
- Lambda层：这是一个神奇的层，看名字就知道它用来把一个函数作用在输入张量上。
这个层可以大大减少你的工作量，当你需要定义的新层的计算不是那么复杂的时候，可以通过lambda层来实现，而不用自己完全重写。
- Highway/Maxout/AtrousConvolution2D层：这个就不多说了，懂的人自然懂，keras还是在一直跟着潮流走的
- Wrapper层：Wrapper层用于将一个普通的层对象进行包装升级，赋予其更多功能。
目前，Wrapper层里有一个TimeDistributed层，用于将普通的层包装为对时间序列输入处理的层，
而Bidirectional可以将输入的递归神经网络层包装为双向的（如把LSTM做成BLSTM）
- Input：补一个特殊的层，Input，这个东西实际上是一个Keras tensor的占位符，
主要用于在搭建Model模型时作为输入tensor使用，这个Input可以通过keras.layers来import。
- stateful与unroll：Keras的递归神经网络层，如SimpleRNN，LSTM等，支持两种特殊的操作。
一种是stateful，设置stateful为True意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。
另一种是unroll，unroll可以将递归神经网络展开，以空间换取运行时间。

Keras的**层对象还有一些有用的属性和方法**，比较有用的是：
- name：别小看这个，从茫茫层海中搜索一个特定的层，
如果你对数数没什么信心，最好是name配合get_layer()来用。
- trainable：这个参数确定了层是可训练的还是不可训练的，
在迁移学习中我们经常需要把某些层冻结起来而finetune别的层，冻结这个动作就是通过设置trainable来实现的。
- input/output：这两个属性是层的输入和输出张量，是Keras tensor的对象，
这两个属性在你需要获取中间层输入输出时非常有用
- get_weights/set_weights：这是两个方法用于手动取出和载入层的参数，
set_weights传入的权重必须与get_weights返回的权重具有同样的shape，
一般可以用get_weights来看权重shape，用set_weights来载入权重

要**在Keras中编写一个自己的层**，需要开一个从Layer（或其他层）继承的类，
除了__init__以为你需要覆盖三个函数：
- build，这个函数用来确立这个层都有哪些参数，哪些参数是可训练的哪些参数是不可训练的。
- call，这个函数在调用层对象时自动使用，里面就是该层的计算逻辑，或计算图了。
显然，这个层的核心应该是一段符号式的输入张量到输出张量的计算过程。
- get_output_shape_for：如果你的层计算后，输入张量和输出张量的shape不一致，
那么你需要把这个函数也重新写一下，返回输出张量的shape，以保证Keras可以进行shape的自动推断

### 优化器、目标函数、初始化策略等等
&emsp;&emsp;objectives是优化目标， 它本质上是一个从张量到数值的函数，
当然，是用符号式编程表达的。
具体的优化目标有mse，mae，交叉熵等等等等，根据具体任务取用即可，当然，也支持自己编写。
需要特别说明的一点是，如果选用categorical_crossentropy作为目标函数，
需要将标签转换为one-hot编码的形式，这个动作通过utils.np_utils.to_categorical来完成。
&emsp;&emsp;optimizers是优化器，没什么可说了，如何选用合适的优化器不在本文讨论范畴。
注意模型是可以传入优化器对象的，你可以自己配置一个SGD，然后将它传入模型中。
另外，最新版本的Keras为所有优化器额外设置了两个参数clipnorm和clipvalue，用来对梯度进行裁剪。
&emsp;&emsp;activation是激活函数，这部分的内容一般不直接使用，而是通过激活层Activation来调用，
此处的激活函数是普通的element-wise激活函数，
如果想使用高级激活函数，请翻到高级激活函数层。
&emsp;&emsp;callback是回调函数，这其实是一个比较重要的模块，回调函数不是一个函数而是一个类，
用于在训练过程中收集信息或进行某种动作。
比如我们经常想画一下每个epoch的训练误差和测试误差，那这些信息就需要在回调函数中收集。
预定义的回调函数中CheckModelpoint，History和EarlyStopping都是比较重要和常用的。
其中:
- CheckPoint用于保存模型
- History记录了训练和测试的信息
- EarlyStopping用于在已经收敛时提前结束训练
- 回调函数LearningRateScheduler支持按照用户的策略调整学习率，做模型精调或研究优化器的同学可能对这个感兴趣。

值得注意的是，**History是模型训练函数fit的返回值**，也就是说即使你没有使用任何回调函数，
找一个变量接住model.fit()，还是能得到不少训练过程中的有用信息。
另外，回调函数还支持将信息发送到远程服务器，以及与Tensorflow的tensorboard联动，
在网页上动态展示出训练和测试的情况（需要使用tensorflow为后端）。
回调函数支持用户自定义，定义方法也非常简单，请参考文档说明编写即可。
另一个文档中没有但实际上有的东西是metrices，这里面定义了一系列用于评价模型的指标，例如“accuracy”。
在训练模型时，可以选择一个或多个指标来衡量模型性能。
### 数据预处理和utils
- to_catgoraical
- convert_kernel: 翻转卷积核
- layer_utils.conver_all_kernels_in_model
### callable
Keras的一大性质是**所有的layer对象都是callable的**。
比如说你想算算一个向量x的sigmoid值是多少，如果用keras的话，你可以这样写：
```Python
import keras.backend as K
from keras.layers import Activation
import numpy as np


x = K.placeholder(shape=(3,))
y = Activation('sigmoid')(x)
f = K.function([x], [y])
out = f([np.array([1, 2, 3])])
```
> 比方说又一次我想给目标函数加一项全变差的正则项，而全变差可以用特定的卷积来实现，
那么我的目标函数的全变差正则项完全就可以用一个Convolution2D层来实现。
把层和模型当做张量的函数来使用，是需要认真贯彻的一个东西。

> 第三行通过调用function函数对计算图进行编译，这个计算图很简单，
就是输入张量经过sigmoid作用编程输出向量，计算图的各种优化通过这一步得以完成，
现在，f就是一个真正的函数了，就可以按照一般的方法使用了。

**模型也是张量到张量的映射，所以Layer是Model的父类**，因此，一个模型本身也可以像上面一样使用。
### Node: Keras的网络层复用
Keras的网络层复用是一个很常用的需求，例如当某一层与多个层相连时，
实际上这层是将同一种计算方式服用多次。
再比如你用一个网络来抽取两条微博的特征，然后在后面用网络来判断两者是否是同一个主题，
那么抽取两次微博的特征这一工作就可以复用同一个网络。

Keras的网络复用有一个叫"Node"，或称"计算节点"的东西来实现。
笼统地说，每当在某一个输入上调用层时，就会为网络层添加一个节点。
这个节点将输入张量映射为输出张量，当你多次调用盖层，就会产生多个结点，结点的下标时0,1,2,3...

当一个层有多个计算节点时，它的input, output, input_shape, output_shape等属性可能是ill-defined的，
因为不清楚你想要的output或input是哪一个。
此时，需要使用get_output_at(), get_input_at(), get_output_shape_at()等以at为后缀结尾的函数，
at的对象就是层的节点编号。例如get_output_shape_at(2)就会返回第3个输出张量的shape。
### Shape与Shape自动推断
input_shape(有时候是input_dim)只需要在模型的首层加以指定。
后面的隔层会根据计算图自动推断。这个功能成为shape的自动推断。
**Keras的自动推断依赖于Layer中的get_output_shape_for函数来实现**。
然而，有时候，这个自动推断会出错。
这种情况发生在一个RNN层后面接Flatten然后又接Dense的时候，这个时候Dense的output_shape无法自动推断出。
这时需要指定RNN的输入序列长度input_length，或者在网络的第一层通过input_shape就指定。
这种情况极少见，大致有个印象即可，遇到的话知道大概是哪里出了问题就好。
### TH与TF的相爱相杀
- dim_ordering
- 卷积层权重的shape
- 卷积层kenel的翻转不翻转问题
