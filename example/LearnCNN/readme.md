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