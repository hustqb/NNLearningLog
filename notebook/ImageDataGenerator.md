参考[Keras中文文档](http://keras-cn.readthedocs.io/en/latest/preprocessing/image/)
# 图片生成器ImageDataGenerator
## 参数
```python
keras.preprocessing.image.ImageDataGenerator(
# 使输入数据集去中心化(均值为0)，按feature执行
featruewise_center=False, 
# 使输入数据的每个样本均值为0
samplewise_center=False, 
# 将输入除以数据集的标准差已完成标准化，按feature
featurewise_std_normalization=False, 
...
)
```
## 方法
`fit(x, augment=False, rounds=1)`：计算依赖于数据的变换所需要的统计信息
（均值方差等），只有使用`featurewise_center`, `featurewise_std_normalization`或
`zca_whitening`时需要次函数。

|parameters|description|
|--|--|
|X|样本数据，秩为4|
|augment|确定是否使用随即提升过的数据|
|round|确定要在数据集上进行多少轮数据提升，默认为1|
|seed|随机数种子|