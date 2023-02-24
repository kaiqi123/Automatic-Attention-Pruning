"""
Refer to https://github.com/kaiqi123/mobileNet-v2_cifar10
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.model_module import PruningModule
from pruning.ModelBuilder import ModelBuilder
# from ModelBuilder import ModelBuilder


"""
Notes:
Org: F.relu6; Now: F.relu
"""

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, builder, input_channel, output_channel, num, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.num = num # new added to print the index of block
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        
        # 1x1   point wise conv
        # self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.conv1 = builder.conv_mobileNet(input_channel, c, kernel_size = 1, padding=0, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1_relu = builder.activation(c)
        
        # 3x3   depth wise conv
        # self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.conv2 = builder.conv_mobileNet(c, c, kernel_size=3, groups=c, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv2_relu = builder.activation(c)
        
        # 1x1   point wise conv
        # self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.conv3 = builder.conv_mobileNet(c, output_channel, kernel_size = 1, padding=0, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        # self.conv3_relu = builder.activation_relu6(output_channel)
        

    def forward(self, inputs):
        
        # x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        # x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
        # print(f"Block: {self.num}")
        x = self.conv1_relu(self.bn1(self.conv1(inputs)))
        # print(f"conv1_relu: {x.shape}")
        x = self.conv2_relu(self.bn2(self.conv2(x)))
        # print(f"conv2_relu: {x.shape}")
        x = self.bn3(self.conv3(x))
        # x = self.conv3_relu(x)
        # print(f"conv3: {x.shape}")

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x



class MobileNetV2(PruningModule):
# class MobileNetV2(nn.Module):
    def __init__(self, builder, output_size, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer 
        # self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv0 = builder.conv_mobileNet(3, int(32*alpha), kernel_size = 3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))
        self.conv0_relu = builder.activation(width=int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        # self.bottlenecks = nn.Sequential(
        self.layer1 = nn.Sequential(
            BaseBlock(builder, 32, 16, num=0, t = 1, downsample = False),
            BaseBlock(builder, 16, 24, num=1, downsample = False),
            BaseBlock(builder, 24, 24, num=2),
            BaseBlock(builder, 24, 32, num=3, downsample = False),
            BaseBlock(builder, 32, 32, num=4),
            BaseBlock(builder, 32, 32, num=5),
            BaseBlock(builder, 32, 64, num=6, downsample = True),
            BaseBlock(builder, 64, 64, num=7),
            BaseBlock(builder, 64, 64, num=8),
            BaseBlock(builder, 64, 64, num=9),
            BaseBlock(builder, 64, 96, num=10, downsample = False),
            BaseBlock(builder, 96, 96, num=11),
            BaseBlock(builder, 96, 96, num=12),
            BaseBlock(builder, 96, 160, num=13, downsample = True),
            BaseBlock(builder, 160, 160, num=14),
            BaseBlock(builder, 160, 160, num=15),
            BaseBlock(builder, 160, 320, num=16, downsample = False))

        # last conv layers and fc layer
        # self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.conv17 = builder.conv_mobileNet(int(320*alpha), 1280, kernel_size = 1, padding=0, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.conv17_relu = builder.activation(1280)

        self.fc3 = nn.Linear(1280, output_size)
        
        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        x = self.conv0(inputs)
        x = self.bn0(x)
        # x = F.relu6(x, inplace = True)
        x = self.conv0_relu(x)
        # print(f"conv0_relu: {x.shape}") # assert x.shape[1:] == torch.Size([32, 32, 32])
        
        # bottlenecks
        x = self.layer1(x)
        # print(f"After bottlenecks: {x.shape}") # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        # x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        x = self.conv17_relu(self.bn1(self.conv17(x)))
        # print(f"conv17_relu: {x.shape}") # assert x.shape[1:] == torch.Size([1280,8,8])
        

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc3(x)

        return x


def build_mobilenetV2(version, num_classes, mask=False, batch_size=128):
    assert version == "mobilenetV2"

    builder = ModelBuilder(mask=mask, batch_size=batch_size)
    model = MobileNetV2(builder, num_classes, alpha = 1)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable_parameters == 2237770 
    
    print("Model Version: {}".format(version))
    print("Num classes: {}".format(num_classes))
    print("Mask: {}".format(mask))
    print("Batch size for relu: {}".format(batch_size))
    print("Trainable number of parameters: {}\n".format(num_trainable_parameters)) # mobilenetV2, cifar10: 2237770
        
    return model


if __name__ == "__main__":

    x = torch.randn(2, 3, 32, 32)
    model = build_mobilenetV2(version="mobilenetV2", num_classes=10, mask=False, batch_size=128)
    y = model(x)
    print(x.shape, y.shape)

    for name, p in model.named_parameters():
        print(name, p.shape)
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_trainable_parameters) # 2.237770M
    assert num_trainable_parameters == 2237770 

