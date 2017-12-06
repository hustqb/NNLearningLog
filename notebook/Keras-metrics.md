# 性能评估
## 使用方法
性能评估模块提供了一系列用于模型性能评估的函数,这些函数在模型编译时由`metrics`关键字设置

性能评估函数类似与目标函数, 只不过该性能的评估结果讲不会用于训练.

可以通过字符串来使用预定义的性能评估函数

也可以自定义一个Theano/TensorFlow函数并使用之
## 可用预定义张量
除fbeta_score额外拥有默认参数beta=1外,其他各个性能指标的参数均为y_true和y_pred

### binary_accuracy
对二分类问题,计算在所有预测值上的平均正确率
```K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)```
$$$$
### categorical_accuracy
对多分类问题,计算再所有预测值上的平均正确率
```K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())```
### sparse_categorical_accuracy
与categorical_accuracy相同,在对稀疏的目标值预测时有用
```K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())```
### top_k_categorical_accracy
计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
```K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)```
### sparse_top_k_categorical_accuracy
与top_k_categorical_accracy作用相同，但适用于稀疏情况
## 定制评估函数
定制的评估函数可以在模型编译时传入,该函数应该以(y_true, y_pred)为参数,并返回单个张量,
或从metric_name映射到metric_value的字典,下面是一个示例:
```
# (y_true, y_pred) as arguments and return a single tensor value.
import keras.backend as K

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)
	
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', mean_pred])
```
## 源码解读
在model.compile中，metrics参数被调用。
> metrics: list of metrics to be evaluated by the model during training and testing.
> Typically you will use `metrics=['accuracy']`.
> To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as `metrics={'output_a': 'accuracy'}`.

然后定义model类的成员变量：
```
self.metrics = metrics
self.metrics_names = ['loss']
self.metrics_tensors = []
```
然后使metrics和output一一对应
> Maps metric functions to model outputs

如果metric中有'accuracy'或者'acc'，重定义:
```
if binary classification or loss=binary_crossentropy：
	metric:=binary_accuracy
elif loss=sparse_categorical_crossentropy:
	metric:=sparse_categorical_accuracy
else:
	metric:=categorical_accuracy
```
最后：
> Adds support for masking to an objective function.
> It trans forms an objective function `fn(y_true, y_pred)` into a cost-masked objective function `fn(y_true, y_pred, mask)`.
