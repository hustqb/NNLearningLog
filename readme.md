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
### LearnResNet
预训练的50层残差网络，识别图片
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
