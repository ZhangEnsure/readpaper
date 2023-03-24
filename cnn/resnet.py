import torch
from torch import nn
from torch.nn import functional as F

class Residual_layer_2(nn.Module):  
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # one of the difference between conv1 and conv2 is the first channel number
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # another difference is that we need to halve the width and depth when 
        # the input_channels is not equal to the num_channels
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # residual block connects the x and y, but both of them have different dims
        # use_1x1conv is the option B mentioned in the paper increasing the computation complex
        # usually used by adding the dimension of x
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        # remember to return relu(Y)
        return F.relu(Y)


class Residual_layer_3(nn.Module):  
    '''
    1x1 卷积降维
    3x3 卷积
    1x1 卷积升维
    '''
    def __init__(self, input_channels, num_channels,output_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()

        # 1x1 conv
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=1)

        # 3x3 conv
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        
        # 1x1 conv
        self.conv3 = nn.Conv2d(num_channels, output_channels, kernel_size=1)
        
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        # remember to return relu(Y)
        return F.relu(Y)
    
'''
定义一个 resnet-34 模型：
1. 首先使用了一个 7x7 卷积.图像高宽减半
2. 随后使用了一个 maxpooling.图像高宽减半
'''
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

'''
ResNet34定义
'''

# we keep the input width, height and channels identical from the first block
# because b1 has already made the input halved twice.
def resnet_block_layer_2(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_layer_2(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual_layer_2(num_channels, num_channels))
    return blk


def resnet_block_layer_3(input_channels, num_channels, output_channels, 
                         num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_layer_3(input_channels, num_channels, output_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual_layer_3(input_channels, num_channels, output_channels,
                                use_1x1conv=False, strides=1))
    return blk


def get_resnet34():
    b2 = nn.Sequential(*resnet_block_layer_2(64, 64, 3, first_block=True))
    b3 = nn.Sequential(*resnet_block_layer_2(64, 128, 4))
    b4 = nn.Sequential(*resnet_block_layer_2(128, 256, 6))
    b5 = nn.Sequential(*resnet_block_layer_2(256, 512, 3))

    ResNet34 = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(), nn.Linear(512, 10))
    return ResNet34


def get_resnet50():
    b2 = nn.Sequential(*resnet_block_layer_3(64,64,256,3,True))
    b3 = nn.Sequential(*resnet_block_layer_3(256,128,512,4,False))
    b4 = nn.Sequential(*resnet_block_layer_3(512,256,1024,6,False))
    b5 = nn.Sequential(*resnet_block_layer_3(1024,512,2048,3,False))

    ResNet50 = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(), nn.Linear(1024, 10))
    return ResNet50