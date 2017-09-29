## 这是一个神经网络的学习笔记
### example
example中是基于keras的神经网络的例子。
#### LearnCNN
采用CNN网络结构的例子

- ImdbCNN 用CNN检测评论是正面的还是负面的
- mnistCNN 用CNN判断手写数字，同时，输出了第一层的输入、输出和权重图片，保存的位置在~/NNLearningLog/example/LearnCNN
#### LearnResNet
这个文件夹可以单独下载，然后在本地的该文件夹下运行py代码，可以检测图片中的物体。
为什么呢？
&emsp;&emsp;因为，我们在这个py文件中定义了ResNet（残差网络），然后使用了预训练好的权重参数（两个h5文件中）。
这样，该文件运行的程序就可以快速的检测输入的图片。图片可以选用任意的jpg格式图片，文件夹下有一个例图：elephant.jpg。
最后的输出由5个名称及其相应的5个概率，分别表示网络识别出的物体名称及其概率，一般来说选最大的。
### famousData
有常用的几个数据集

- imdb.npz和imdb_word_index.json
- mnist.npz
- cifar-10-batches
### mnist_first_output
这是example/LearnCNN/mnistCNN.py的运行结果：
	
	1. 0(1, 2, 3, 4)inter.png分别表示mnistCNN网络第一层（第一个卷积层）的输出
	2. 0(1, 2, 3, 4)org.png分别表示mnistCNN网络第一层的输入
	3. weight.png表示mnistCNN网络输入层到第一层的权重（卷积核参数）
### notebook
笔记
- 知乎大神的AI经验
- 学习资料
- Softmax变换
- 数据归一化
- Keras学习笔记
- 神经网络为什么要归一化
