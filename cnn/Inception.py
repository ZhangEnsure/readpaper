import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, List, Optional, Tuple

'''
Inception结构图
https://www.yuque.com/shuoouba/deeplearning/lq5ib9zqrbqlaa72
'''




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
            self.branch3x3dbl_2(self.branch3x3dbl_3(x)))
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
        branch1x3x1dbl = self.branch1x3x1dbl_3(
            self.branch1x3x1dbl_2(self.branch1x3x1dbl_1(x)))
        branch3x3dbl = self.branch3x3dbl_4(self.branch3x3dbl_3(
            self.branch3x3dbl_2(self.branch3x3dbl_1(x))))
        return [branch1x1, branch_pool, branch1x3x1dbl, branch3x3dbl]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(self._forward(x), dim=1)