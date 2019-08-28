import torch.nn as nn
from .base import PyTorchModel

from torchvision.models.resnet import BasicBlock


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class CNN(nn.Module, PyTorchModel):
    def __init__(self, hidden_1_size=64, hidden_block_size=64,
                 input_channels=3, output_size=17, blocks=2):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_1_size,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_1_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        blocks_ = []
        # Fix feature map size
        self.conv2 = conv1x1(hidden_1_size, hidden_block_size)
        for block in range(blocks):
            blocks_.append(BasicBlock(hidden_block_size,
                                      hidden_block_size, stride=1))
        self.blocks = nn.Sequential(*blocks_)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_block_size, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        output = self.fc(x.view(x.size(0), -1))
        return output
