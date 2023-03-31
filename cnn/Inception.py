import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, List, Optional, Tuple

# pip install torchsummary
from torchsummary import summary

'''
Inception结构图
https://www.yuque.com/shuoouba/deeplearning/lq5ib9zqrbqlaa72
'''


class Inception3(nn.Module):
    '''
    没有附加分类器
    '''

    def __init__(
            self,
            num_classes: int = 1000,
            dropout: float = 0.5
    ) -> None:
        super().__init__()
        inception_blocks = [BasicConv2d, InceptionA,
                            InceptionB, InceptionC, InceptionD, InceptionE]
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)

        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, c7=128)
        self.Mixed_6c = inception_c(768, c7=160)
        self.Mixed_6d = inception_c(768, c7=160)
        self.Mixed_6e = inception_c(768, c7=192)

        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)

        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)

        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)

        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x


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


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int,
                 conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.branch_pool_1x1 = conv_block(
            in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_1(x)))
        branch_pool = self.branch_pool_1x1(self.branch_pool(x))
        return [branch1x1, branch5x5, branch3x3, branch_pool]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(self._forward(x), dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_3(
            self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch_pool = self.branch_pool(x)

        return [branch3x3, branch_pool, branch3x3dbl]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(self._forward(x), dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, c7: int,
                 conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        self.branch2x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch2x7dbl_2 = conv_block(
            c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch2x7dbl_3 = conv_block(
            c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch4x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch4x7dbl_2 = conv_block(
            c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch4x7dbl_3 = conv_block(
            c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch4x7dbl_4 = conv_block(
            c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch4x7dbl_5 = conv_block(
            c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.branch_pool_conv = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)
        branch2x7dbl = self.branch2x7dbl_3(
            self.branch2x7dbl_2(self.branch2x7dbl_1(x)))
        branch4x7dbl = self.branch4x7dbl_5(self.branch4x7dbl_4(
            self.branch4x7dbl_3(self.branch4x7dbl_2(self.branch4x7dbl_1(x)))))
        branch_pool = self.branch_pool_conv(self.branch_pool(x))
        return [branch1x1, branch2x7dbl, branch4x7dbl, branch_pool]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(self._forward(x), dim=1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch73dbl_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch73dbl_2 = conv_block(
            192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch73dbl_3 = conv_block(
            192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch73dbl_4 = conv_block(192, 192, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_2(self.branch3x3_1(x))
        branch73dbl = self.branch73dbl_4(self.branch73dbl_3(
            self.branch73dbl_2(self.branch73dbl_1(x))))
        branch_pool = self.branch_pool(x)
        return [branch3x3, branch73dbl, branch_pool]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(self._forward(x), dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = conv_block(in_channels, 192, kernel_size=1)

        self.branch1x3x1dbl_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch1x3x1dbl_2 = conv_block(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch1x3x1dbl_3 = conv_block(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_4 = conv_block(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch_pool = self.branch_pool_conv(self.branch_pool(x))

        branch1x3x1dbl_1 = self.branch1x3x1dbl_3(self.branch1x3x1dbl_1(x))
        branch1x3x1dbl_2 = self.branch1x3x1dbl_2(self.branch1x3x1dbl_1(x))

        branch3x3dbl_1 = self.branch3x3dbl_4(
            self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch3x3dbl_2 = self.branch3x3dbl_3(
            self.branch3x3dbl_2(self.branch3x3dbl_1(x)))

        return [branch1x1, branch_pool, branch1x3x1dbl_1, branch1x3x1dbl_2, branch3x3dbl_1, branch3x3dbl_2]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(self._forward(x), dim=1)


x = torch.rand(size=(10, 3, 299, 299))
net = Inception3().cuda()
# print(net(x))
# summary(model=net, input_size=(3,168, 168), batch_size=10, device="cuda")