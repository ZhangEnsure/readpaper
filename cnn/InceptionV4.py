import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, List, Optional, Tuple


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        # 不经过复制操作，而是直接在原来的内存上改变它的值
        return F.relu(x, inplace=True)


class stem(nn.Module):
    def __init__(self, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.stem1 = stem_1()
        self.stem2 = stem_2()
        self.conv = conv_block(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def _forward(self, x):
        x = self.stem2(self.stem1(x))
        conv = self.conv(x)
        maxpool = self.maxpool(x)
        return [conv, maxpool]

    def forward(self, x):
        return torch.cat(self._forward(x), dim=1)


class stem_1(nn.Module):
    def __init__(self, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv1 = conv_block(3, 32, kernel_size=3, stride=2)
        self.conv2 = conv_block(32, 32, kernel_size=3)
        self.conv3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = conv_block(64, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        maxpooling = self.maxpooling(x)
        conv4 = self.conv4(x)
        return [maxpooling, conv4]

    def forward(self, x):
        return torch.cat(self._forward(x), dim=1)


class stem_2(nn.Module):
    def __init__(self, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1db1 = conv_block(160, 64, kernel_size=1)
        self.branch1db2 = conv_block(64, 96, kernel_size=3)

        self.branch2db1 = conv_block(160, 64, kernel_size=1)
        self.branch2db2 = conv_block(
            64, 64, kernel_size=(7, 1), padding=(3, 0))
        self.branch2db3 = conv_block(
            64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.branch2db4 = conv_block(64, 96, kernel_size=3)

    def _forward(self, x):
        branch1 = self.branch1db2(self.branch1db1(x))
        branch2 = self.branch2db4(self.branch2db3(
            self.branch2db2(self.branch2db1(x))))
        return [branch1, branch2]

    def forward(self, x):
        return torch.cat(self._forward(x), dim=1)


class Inception_resnet_A(nn.Module):
    def __init__(self, in_features: int, conv_block: Optional[Callable[..., nn.Module]] = None, scale: float = 0.1) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.scale = scale

        self.branch1 = conv_block(
            in_channels=in_features, out_channels=32, kernel_size=1)

        self.branch2db1 = conv_block(
            in_channels=in_features, out_channels=32, kernel_size=1)
        self.branch2db2 = conv_block(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.branch3db1 = conv_block(
            in_channels=in_features, out_channels=32, kernel_size=1)
        self.branch3db2 = conv_block(
            in_channels=32, out_channels=48, kernel_size=3, padding=1)
        self.branch3db3 = conv_block(
            in_channels=48, out_channels=64, kernel_size=3, padding=1)

        self.branch1x1cat = nn.Conv2d(
            in_channels=128, out_channels=384, kernel_size=1)

        if in_features != 384:
            self.use_1x1conv = nn.Conv2d(
                in_channels=in_features, out_channels=384, kernel_size=1, bias=False)
        else:
            self.use_1x1conv = None

    def _forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2db2(self.branch2db1(x))
        branch3 = self.branch3db3(self.branch3db2(self.branch3db1(x)))
        x = torch.cat([branch1, branch2, branch3], dim=1)
        return self.branch1x1cat(x)

    def forward(self, x: Tensor) -> Tensor:
        Y = self._forward(x)
        # 在残差连接前的最后一个线性层之后乘以一个缩小系数
        Y = Y * self.scale
        # 如果 x 与 Y 维度不匹配，则变化 x 的通道数
        if self.use_1x1conv:
            x = self.use_1x1conv(x)
        return F.relu(Y+x)


class Redection_A_resnet_v2(nn.Module):
    def __init__(self, in_features: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2 = conv_block(
            in_channels=in_features, out_channels=384, kernel_size=3, stride=2)

        self.branch3db1 = conv_block(
            in_channels=in_features, out_channels=256, kernel_size=1)
        self.branch3db2 = conv_block(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.branch3db3 = conv_block(
            in_channels=256, out_channels=384, kernel_size=3, stride=2)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3db3(self.branch3db2(self.branch3db1(x)))
        return [branch1, branch2, branch3]

    def forward(self, x):
        return torch.cat(self._forward(x), dim=1)


class Inception_resnet_B(nn.Module):
    def __init__(self, in_features: int, conv_block: Optional[Callable[..., nn.Module]] = None, scale: float = 0.1) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.scale = scale

        self.branch1db1 = conv_block(
            in_channels=in_features, out_channels=192, kernel_size=1)

        self.branch2db1 = conv_block(
            in_channels=in_features, out_channels=128, kernel_size=1)
        self.branch2db2 = conv_block(
            in_channels=128, out_channels=160, kernel_size=(1, 7), padding=(0, 3))
        self.branch2db3 = conv_block(
            in_channels=160, out_channels=192, kernel_size=(7, 1), padding=(3, 0))

        self.branch1x1cat = nn.Conv2d(
            in_channels=384, out_channels=1154, kernel_size=1)

        self.use_1x1conv = None
        if in_features != 1154:
            self.use_1x1conv = nn.Conv2d(
                in_channels=in_features, out_channels=1154, kernel_size=1, bias=False)

    def _forward(self, x):
        branch1 = self.branch1db1(x)
        branch2 = self.branch2db3(self.branch2db2(self.branch2db1(x)))
        branch = torch.cat([branch1, branch2], dim=1)
        return self.branch1x1cat(branch)

    def forward(self, x):
        Y = self._forward(x) * self.scale
        if self.use_1x1conv:
            x = self.use_1x1conv(x)
        return F.relu(Y+x)


class Redection_B_resnet_v2(nn.Module):
    def __init__(self, in_features: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2db1 = conv_block(
            in_channels=in_features, out_channels=256, kernel_size=1)
        self.branch2db2 = conv_block(
            in_channels=256, out_channels=384, kernel_size=3, stride=2)

        self.branch3db1 = conv_block(
            in_channels=in_features, out_channels=256, kernel_size=1)
        self.branch3db2 = conv_block(
            in_channels=256, out_channels=288, kernel_size=3, stride=2)

        self.branch4db1 = conv_block(
            in_channels=in_features, out_channels=256, kernel_size=1)
        self.branch4db2 = conv_block(
            in_channels=256, out_channels=288, kernel_size=3, padding=1)
        self.branch4db3 = conv_block(
            in_channels=288, out_channels=320, kernel_size=3, stride=2)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2db2(self.branch2db1(x))
        branch3 = self.branch3db2(self.branch3db1(x))
        branch4 = self.branch4db3(self.branch4db2(self.branch4db1(x)))
        return [branch1, branch2, branch3, branch4]

    def forward(self, x):
        return torch.cat(self._forward(x), dim=1)


class Inception_resnet_C(nn.Module):
    def __init__(self, in_features: int, conv_block: Optional[Callable[..., nn.Module]] = None, scale: float = 0.1) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.scale = scale

        self.branch1db1 = conv_block(
            in_channels=in_features, out_channels=192, kernel_size=1)

        self.branch2db1 = conv_block(
            in_channels=in_features, out_channels=192, kernel_size=1)
        self.branch2db2 = conv_block(
            in_channels=192, out_channels=224, kernel_size=(1, 3), padding=(0, 1))
        self.branch2db3 = conv_block(
            in_channels=224, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch1x1conv = conv_block(in_channels=448, out_channels=2048, kernel_size=1)

        self.use_1x1conv = None
        if in_features != 2048:
            self.use_1x1conv = nn.Conv2d(
                in_channels=in_features, out_channels=2048, kernel_size=1, bias=False)

    def _forward(self, x):
        branch1 = self.branch1db1(x)
        branch2 = self.branch2db3(self.branch2db2(self.branch2db1(x)))
        branch = torch.cat([branch1, branch2], dim=1)
        return self.branch1x1conv(branch)

    def forward(self, x):
        Y = self._forward(x)*self.scale
        if self.use_1x1conv:
            x = self.use_1x1conv(x)
        return F.relu(Y+x)


class Inception_resnet_v2(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.stem = stem()
        self.inceptionA_a = Inception_resnet_A(384)
        self.inceptionA_b = Inception_resnet_A(384)
        self.inceptionA_c = Inception_resnet_A(384)
        self.inceptionA_d = Inception_resnet_A(384)
        self.inceptionA_e = Inception_resnet_A(384)

        self.reductionA = Redection_A_resnet_v2(384)

        self.inceptionB_1 = Inception_resnet_B(1152)
        self.inceptionB_2 = Inception_resnet_B(1154)
        self.inceptionB_3= Inception_resnet_B(1154)
        self.inceptionB_4 = Inception_resnet_B(1154)
        self.inceptionB_5 = Inception_resnet_B(1154)
        self.inceptionB_6 = Inception_resnet_B(1154)
        self.inceptionB_7 = Inception_resnet_B(1154)
        self.inceptionB_8 = Inception_resnet_B(1154)
        self.inceptionB_9 = Inception_resnet_B(1154)
        self.inceptionB_10 = Inception_resnet_B(1154)

        self.reductionB = Redection_B_resnet_v2(1154)

        self.inceptionC_1 = Inception_resnet_C(2146)
        self.inceptionC_2 = Inception_resnet_C(2048)
        self.inceptionC_3 = Inception_resnet_C(2048)
        self.inceptionC_4 = Inception_resnet_C(2048)
        self.inceptionC_5 = Inception_resnet_C(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # n x 3 x 299 x 299
        x = self.stem(x)
        # n x 384 x 35 x 35
        x = self.inceptionA_a(x)
        # n x 384 x 35 x 35
        x = self.inceptionA_b(x)
        # n x 384 x 35 x 35
        x = self.inceptionA_c(x)
        # n x 384 x 35 x 35
        x = self.inceptionA_d(x)
        # n x 384 x 35 x 35
        x = self.inceptionA_e(x)
        # n x 384 x 35 x 35
        x = self.reductionA(x)
        # n x 1152 x 17 x 17
        x = self.inceptionB_1(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_2(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_3(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_4(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_5(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_6(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_7(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_8(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_9(x)
        # n x 1154 x 17 x 17
        x = self.inceptionB_10(x)
        # n x 1154 x 17 x 17
        x = self.reductionB(x)
        # n x 2146 x 8 x 8
        x = self.inceptionC_1(x)
        # n x 2048 x 8 x 8
        x = self.inceptionC_2(x)
        # n x 2048 x 8 x 8
        x = self.inceptionC_3(x)
        # n x 2048 x 8 x 8
        x = self.inceptionC_4(x)
        # n x 2048 x 8 x 8
        x = self.inceptionC_5(x)
        # n x 2048 x 8 x 8
        x = self.avgpool(x)
        # n x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 10 (num_classes)
        return x


x = torch.rand(size=(10,3,299,299))
net = Inception_resnet_v2()
x = net(x)
print(x.shape)

'''
Redection_A = Redection_A_resnet_v2(384)
x = torch.rand(size=(10, 384, 35, 35))
print(Redection_A(x).shape)
x = Redection_A(x)
inception_resnet_b1 = Inception_resnet_B(in_features=1152)
inception_resnet_b2 = Inception_resnet_B(in_features=1154)
x = inception_resnet_b1(x)
print(x.shape)
x = inception_resnet_b2(x)
print(x.shape)
Redection_B = Redection_B_resnet_v2(1154)
x = Redection_B(x)
print(x.shape)
inception_resnet_c1 = Inception_resnet_C(2146)
x = inception_resnet_c1(x)
print(x.shape)
inception_resnet_c2 = Inception_resnet_C(2048)
x = inception_resnet_c2(x)
print(x.shape)
'''

'''
# 测试 inception-resnet-a
inception_resnet_a1 = Inception_resnet_A(in_features=256)
inception_resnet_a2 = Inception_resnet_A(in_features=384)
x = torch.rand(size=(10,256,35,35))
x = inception_resnet_a1(x)
print(x.shape)
x = inception_resnet_a2(x)
print(x.shape)
'''


# 测试主干网络
# stem = stem()
# x = torch.rand(size=(10,3,299,299))
# print('stem处理之后：'+str(stem(x).shape))

