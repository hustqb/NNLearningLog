__main__.py存在的意义：
在本地运行的时候，Python包间调用没有问题，但是当代码转移到服务器上跑时，报错`no module`。
无奈去网上找的一个解决办法：
1. 添加一个`__main__.py`
2. 修改py文件内部的import路径
3. 运行命令从`python xxx.py`改为`python -m 绝对路径xxx`.例如
> python -m example/LearnCNN/mnistCNN