# 演示keras的特性
## antirectifier
演示keras中自定义网络中间层。
自定义层是一个类，继承自`keras.layers.layer`，首先要重写`call()`函数，
然后，如果改变了tensor shape的大小，则还需重写`compute_output_shape()`函数。

本示例中自定义的Antirectifier中间层，用来替代ReLU激活函数，可以用较小的网络实现较高的精度。