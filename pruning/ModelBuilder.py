import torch.nn as nn
import torch.nn.functional as F
import math

from pruning.model_module import PruningModule, MaskedLinear, MaskedConv2d, MaskedConv2d_MobileNet
from pruning.model_module import CustomizedRelu, CustomizedRelu6

"""
vgg, mobilenet, and shufflenet use this 
mobilenet and shufflenet use conv_mobileNet for all conv layers
"""
class ModelBuilder(object):
    def __init__(self, mask, batch_size):
        self.mask = mask
        self.batch_size = batch_size
    
    def conv_mobileNet(self, in_planes, out_planes, kernel_size, groups=1, stride=1, padding=1, bias=False):
        """
        Difference with nn.Conv2d:
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        """
        conv = MaskedConv2d_MobileNet if self.mask else nn.Conv2d
        conv = conv(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        return conv

    def conv(self, in_planes, out_planes, kernel_size, groups=1, stride=1, padding=1, bias=False):
        """
        Difference with nn.Conv2d:
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        """
        conv = MaskedConv2d if self.mask else nn.Conv2d
        conv = conv(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        return conv

    def conv3x3(self, in_planes, out_planes, bias=False, stride=1):
        return self.conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
    
    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(in_planes, out_planes, kernel_size=1, stride=stride)
        return c

    def activation(self, width=None):
        activation = CustomizedRelu(width, batch_size=self.batch_size, inplace=True) if self.mask else nn.ReLU(inplace=True)
        return activation

    def activation_relu6(self, width=None):
        activation = CustomizedRelu6(width, batch_size=self.batch_size, inplace=True) if self.mask else nn.ReLU(inplace=True)
        return activation

    def fc(self, in_planes, out_planes):
        return MaskedLinear(in_planes, out_planes) if self.mask else nn.Linear(in_planes, out_planes)