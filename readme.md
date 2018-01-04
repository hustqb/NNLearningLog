# 这是一个神经网络的学习笔记
## example
example中是基于keras的神经网络的例子。
### hdf5
预训练的网络模型
### LearnCNN
采用CNN网络结构的例子

- ImdbCNN 用CNN检测评论是正面的还是负面的
- mnistCNN 用CNN判断手写数字，同时，输出了第一层的输入、输出和权重图片，保存的位置在~/NNLearningLog/example/LearnCNN
- cifar_10_cnn 用CNN对图片进行分类。
- vanGogh 图像风格转移
### LearnGAN
- mnist_acgan 
- mnist_dcgan  deep convlutional generative adversarial network
### LearnLSTM
- imdb_cnn_lstm CNN与LSTM的结合，很简单的例子
	1. 填充样本序列
	2. 嵌入层+Dropout+一维卷积+池化层+LSTM+输出层
- imdb_lstm LSTM网络，简单
	1. 填充样本序列
	2. 嵌入层+LSTM+输出层
### LearnResNet
预训练的50层残差网络，识别图片
### LearnRNN
- addition_rnn 自动计算1000以内的加法运算, 非常巧妙的利用seq2seq思想
	1. 处理数据
	2. 网络结构：
		1. 第一层LSTM提取输入(n, 7, 12)的隐变量(n, 128)；
		2. 第二层RepeatVector将隐变量分成4份(n, 4, 128)；
		3. 第三层LSTM(return_sequences=True)解码这4个隐变量(n, 4, 128)；
		5. 第四层TimeDistributed+Dense层，用4个相同的全连接层输出解码后的隐变量，输出(n, 4, 12).
- babi_rnn 阅读理解，输入一段话和一个问题，输出答案
- mnist_hierarchical_rnn 分层的rnn
	1. 运用TimeDistributed对原图像(n, 28, 28, 3)每一行调用lstm得(n, 28, 128)
	2. 再来一个lstm得(n, 128)，最后接一个分类输出Dense
### siamese
图片对比：输入两个图片，判断图片内容是否相同。该网络常用于人脸识别。
### utils
工具库
### variationalAutoencode
变分自编码器，常用于创建样本。
## famousData
有常用的几个数据集

- imdb.npz和imdb_word_index.json
- mnist.npz
- cifar-10-batches
## mnist_first_output
这是example/LearnCNN/mnistCNN.py的运行结果：
	
	1. 0(1, 2, 3, 4)inter.png分别表示mnistCNN网络第一层（第一个卷积层）的输出
	2. 0(1, 2, 3, 4)org.png分别表示mnistCNN网络第一层的输入
	3. weight.png表示mnistCNN网络输入层到第一层的权重（卷积核参数）
## notebook
笔记
- *ImageDataGenerator* 是keras中的一个图片预处理API
- Keras函数式API
- 知乎大神的AI经验
- 学习资料
- Softmax变换
- 数据归一化
- Keras学习笔记
- 神经网络为什么要归一化
## vanGoghFigure
图片风格转移的运行结果
