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