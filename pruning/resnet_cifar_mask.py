import math
import torch
import torch.nn as nn
import numpy as np
import sys

from pruning.model_module import PruningModule, MaskedLinear, MaskedConv2d
from pruning.model_module import CustomizedRelu

class ResNetCifarBuilder(object):
    def __init__(self, mask, batch_size):
        self.mask = mask
        self.batch_size = batch_size
    
    def conv(self, in_planes, out_planes, kernel_size, groups=1, stride=1, padding=1, bias=False):
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

    def conv3x3(self, in_planes, out_planes, stride=1):
        # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        return self.conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
    def activation(self, width=None):
        activation = CustomizedRelu(width, batch_size=self.batch_size, inplace=True) if self.mask else nn.ReLU(inplace=True)
        return activation

    def fc(self, in_planes, out_planes):
        return MaskedLinear(in_planes, out_planes) if self.mask else nn.Linear(in_planes, out_planes)

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1_relu = builder.activation(width=planes)
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2_relu = builder.activation(width=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv1_relu(out)
        # print(f"conv1_relu.shape: {x.shape}") 

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.conv2_relu(out)
        # print(f"conv2_relu.shape: {x.shape}") 

        return out

class ResNet_Cifar(PruningModule):
# class ResNet_Cifar(nn.Module):

    def __init__(self, builder, block, layers, num_classes):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = builder.conv3x3(3, 16, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1_relu = builder.activation(width=16)
        self.layer1 = self._make_layer(builder, block, 16, layers[0])
        self.layer2 = self._make_layer(builder, block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 64, layers[2], stride=2)
                
        if len(layers) == 4:
            self.layer4 = self._make_layer(builder, block, 128, layers[3], stride=2)            
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.fc3 = builder.fc(128 * block.expansion, num_classes)
        else:
            self.avgpool = nn.AvgPool2d(8, stride=1)
            # self.fc = nn.Linear(64 * block.expansion, num_classes)
            self.fc3 = builder.fc(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"x.shape: {x.shape}") # (128, 3, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv1_relu(x)
        # print(f"conv1_relu.shape: {x.shape}") # (128, 16, 32, 32)
        
        # print("layer1")
        x = self.layer1(x)
        # print("layer2")
        x = self.layer2(x)
        # print("layer3")
        x = self.layer3(x)

        if 'layer4' in self._modules:
            # print("layer4")
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x



resnet_versions = {
    "resnet18Cifar": {
        "net": ResNet_Cifar,
        "block": BasicBlock,
        "layers": [3, 3, 3],
    },
    "resnet56Cifar": {
        "net": ResNet_Cifar,
        "block": BasicBlock,
        "layers": [9, 9, 9],
    },
    "resnet50Cifar": {
        "net": ResNet_Cifar,
        "block": BasicBlock,
        "layers": [3, 4, 6, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
}

def build_resnet(version, config, num_classes, verbose=True, mask=False, batch_size=128):
    # config is useless
    version = resnet_versions[version]
    # config = resnet_configs[config]

    builder = ResNetCifarBuilder(mask=mask, batch_size=batch_size)
    if verbose:
        print("Version: {}".format(version))
        # print("Config: {}".format(config))
        print("Num classes: {}".format(num_classes))
        print("Mask: {}".format(mask))
        print("Batch size for relu: {}\n".format(batch_size))
    
    model = version["net"](
        builder=builder,
        block=version["block"], 
        layers=version["layers"], 
        num_classes=num_classes,
    )
    return model



if __name__ == '__main__':
    
    # model_version = "resnet18Cifar"
    # model_version = "resnet56Cifar"
    model_version = "resnet50Cifar" #1334618

    model = build_resnet(model_version, "fanin", num_classes=10, mask=False, batch_size=128)

    y = model(torch.randn(1, 3, 32, 32))
    print(model)
    print(y.size())

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_trainable_parameters)
    if model_version == "resnet18_cifar":
        assert num_trainable_parameters == 272474
    elif model_version == "resnet56Cifar":
        assert num_trainable_parameters == 855770

    for name, p in model.named_parameters():
        print(name, p.shape)
