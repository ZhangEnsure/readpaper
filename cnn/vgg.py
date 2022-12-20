import torch
import torch.nn as nn

from typing import cast, Dict, List, Optional, Union


class VGG(nn.Module):
    def __init__(self, features: nn.Module, init_weights: bool = False, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 最后一层 conv2d 都是 512 通道数
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
        # 是否选择在构造 model 时进行初始化
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layer: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 把 v 值转换为 int 类型
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 把 [] 解包传入 nn.Sequential 中
    return nn.Sequential(*layer)


def vgg11(init_weights: bool = False, num_classes: int = 1000, dropout: float = 0.5):
    features = make_layers(cfgs["A"], batch_norm=False)
    model = VGG(features, init_weights, num_classes, dropout)
    return model


def vgg13(init_weights: bool = False, num_classes: int = 1000, dropout: float = 0.5):
    features = make_layers(cfgs["B"], batch_norm=False)
    model = VGG(features, init_weights, num_classes, dropout)
    return model


def vgg16(init_weights: bool = False, num_classes: int = 1000, dropout: float = 0.5):
    features = make_layers(cfgs["D"], batch_norm=False)
    model = VGG(features, init_weights, num_classes, dropout)
    return model

    
def vgg19(init_weights: bool = False, num_classes: int = 1000, dropout: float = 0.5):
    features = make_layers(cfgs["E"], batch_norm=False)
    model = VGG(features, init_weights, num_classes, dropout)
    return model