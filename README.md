# 阅读论文的记录

论文一读就忘，虽有标注，但是还是记下来为妙。

## 计划的更新

从经典的 CNN 卷积神经网络出发：

| 网络结构 |                                       学习记录                                       | 时间       |
| :-------: | :----------------------------------------------------------------------------------: | ---------- |
|  AlexNet  |    [语雀-AlexNet](https://www.yuque.com/shuoouba/deeplearning/syg020gvb5m1c2k9 "None")    | 2022-12-18 |
|    VGG    |                                         未更                                         |            |
|    NIN    | [语雀-Network in Network](https://www.yuque.com/shuoouba/deeplearning/hwp2vtzcn5oo0abm) | 2023-03-02 |
| GoogLeNet |                                         未更                                         |            |
|  ResNet  |                                         未更                                         |            |

轻量化 CNN 网络

|   网络结构   | 学习记录 | 时间 |
| :----------: | :------: | ---- |
| MobileNet V1 |   未更   |      |
| MobileNet V2 |   未更   |      |

## 使用说明

train.ipynb 使用 pytorch 作为框架，安装好 pytorch 即可运行。

此外，train.ipynb 使用的函数定义在 utils/tools.py 和 utils/loadData.py 中。utils/tools.py 是根据李沐老师 d2l/pytorch 代码进行改写得到的，在此非常感谢李沐老师团队的高质量代码。需要注意的是，如果使用 pytorch 更改数据类型、更改 torch.tensor size 等操作和 numpy 包对 numpy.array 数组操作存在一定的差异，这给当时改写代码带来了一些需要学习的内容。utils/loadData.py 分别定义了可以加载两个数据集的函数，一个是 Fashion-MNIST，另一个是 CIFAR-10 数据集。这两个数据集下载后保存路径默认为 readpaper 同级目录下的 data 目录。

这两个数据集最大的不同是图像的通道数。Fashion-MNIST 是灰度图，只有一个通道；CIFAR-10 是彩色图，有三个通道。所以在使用不同模型的时候，需要在 cnn/model.py 修改第一个 conv2d 卷积层的输入通道数，否则会运行失败。

模型运行的超参数可以在 utils/config.py 中进行修改，这个超参数文件对于小项目来说其实挺麻烦的，一旦运行 train.ipynb 文件就会把导入的文件装入，所以一旦我们想修改 config.py 中的超参数就需要重启内核。
