'''
Refer to "CONTRASTIVE REPRESENTATION DISTILLATION"
'''
import torch.nn as nn
import torch.nn.functional as F
import math

from pruning.model_module import PruningModule, MaskedLinear, MaskedConv2d
from pruning.model_module import CustomizedRelu
from pruning.ModelBuilder import ModelBuilder


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class BLOCK(nn.Module):

    def __init__(self, builder, in_channels, out_channels, batch_norm=False):
        super(BLOCK, self).__init__()
        self.conv1 = builder.conv3x3(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1_relu = builder.activation(width=out_channels)
        self.batch_norm = batch_norm
    
    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn(out)
        out = self.conv1_relu(out)
        return out


class VGG(PruningModule):

    def __init__(self, cfg, batch_norm=False, builder=None, num_classes=1000):
        super(VGG, self).__init__()
        self.builder = builder
        self.layer1 = self._make_layers(cfg[0], batch_norm, 3)
        self.layer2 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.layer3 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.layer4 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.layer5 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc3 = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        h = x.shape[2]
        x = self.layer1(x) 

        x = self.pool1(x)
        x = self.layer2(x) 

        x = self.pool2(x)
        x = self.layer3(x)

        x = self.pool3(x)
        x = self.layer4(x) 

        if h == 64:
            x = self.pool4(x)
        x = self.layer5(x) 

        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x
    
    def _make_layers(self, cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                raise EOFError("v is never equal to M")
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(BLOCK(self.builder, in_channels, v, batch_norm=batch_norm))
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


vgg_model_dict = {
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
}

def build_vgg(version, num_classes, mask=False, batch_size=128):
    assert version in vgg_model_dict.keys()

    builder = ModelBuilder(mask=mask, batch_size=batch_size)
    model = vgg_model_dict[version](builder=builder, num_classes=num_classes)
    
    print("Model Version: {}".format(version))
    print("Num classes: {}".format(num_classes))
    print("Mask: {}".format(mask))
    print("Batch size for relu: {}\n".format(batch_size))
        
    return model


if __name__ == '__main__':
    import torch

    version = "vgg16"
    num_classes = 10
    model = build_vgg(version, num_classes, mask=False, batch_size=128)
    
    x = torch.randn(2, 3, 32, 32)
    logit = model(x)
    print(logit.shape)

    for name, p in model.named_parameters():
        print(name, p.shape)
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_trainable_parameters)
