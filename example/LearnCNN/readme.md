## 注意
### mnistCNN
mnistCNN中的数据来自于`NNLearningLog/famousData/mnist.npz`。
同时在代码中还调用了包外`NNLearningLog/example/utils/datasets.py`中的`load_mnist`。

当**涉及到包间模块的调用问题**时，如果你是使用的IDE，那应该可以识别。
但是，如果是在命令行中运行，就需要一点改进了。
改进参考[这个博客](http://blog.csdn.net/luo123n/article/details/49849649):

1.在`example`中新建空文件`__main__.py`
2.在命令行中运行时，切换到`NNLearnLog`目录下，
使用命令`python -m example/LearnCNN/mnistCNN`
### ImdbCNN
ImdbCNN中的数据是有关于评论的数据，来自于`NNLearningLog/famousData/imdb.npz`
同时在代码中还调用了包外`NNLearningLog/example/utils/datasets.py`中的`load_imdb`。

虽然imdb数据是时间序列，但它用CNN网络来训练可以达到80%的准确率。
不过，如果把优化器从Adam换成sgd，网络会没有学习能力，只有50%的准确率。
### cifar_10_cnn
cifar_10_cnn的数据是图片，它的工作就是将图片分类成10类。
数据的获取方式和上面的几个例程一样。

用简单的CNN-maxpool-dense来做，准确率为60%，不过据说一直迭代最终可以达到80%的准确率。
### vanGogh
图像风格转移，通过图片A的内容和图片B的风格产生一个新的图片C。
比如，我们可以让一个普通的图片加上梵高的作品风格，生成一个新的艺术作品。
这个技术的关键在于构建损失函数，损失函数包括三部分：
1. total variation loss 表示图片C中像素的连续性；
2. style loss
3. content loss
参考论文：[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)