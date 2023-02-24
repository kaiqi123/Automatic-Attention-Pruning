'''ShuffleNetV2 in PyTorch.
See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.model_module import PruningModule
from pruning.ModelBuilder import ModelBuilder


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, builder, in_channels, split_ratio=0.5, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv1 = builder.conv_mobileNet(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1_relu = builder.activation(width=in_channels)

        # self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.conv2 = builder.conv_mobileNet(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv3 = builder.conv_mobileNet(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.conv3_relu = builder.activation(width=in_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        # out = F.relu(self.bn1(self.conv1(x2)))
        out = self.conv1_relu(self.bn1(self.conv1(x2)))
        # print(f"BasicBlock, conv1_relu: {out.shape}")

        out = self.bn2(self.conv2(out))
        # print(f"BasicBlock, bn2: {out.shape}")

        preact = self.bn3(self.conv3(out))
        # out = F.relu(preact)
        out = self.conv3_relu(preact)
        # print(f"BasicBlock, conv3_relu: {out.shape}")

        preact = torch.cat([x1, preact], 1)
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        if self.is_last:
            return out, preact
        else:
            return out


class DownBlock(nn.Module):
    def __init__(self, builder, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.conv1 = builder.conv_mobileNet(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.conv2 = builder.conv_mobileNet(in_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2_relu = builder.activation(width=mid_channels)
        # right
        # self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.conv3 = builder.conv_mobileNet(in_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3_relu = builder.activation(width=mid_channels)

        # self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.conv4 = builder.conv_mobileNet(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        # self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.conv5 = builder.conv_mobileNet(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        self.conv5_relu = builder.activation(width=mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        # print(f"DownBlock, bn1: {out1.shape}")

        # out1 = F.relu(self.bn2(self.conv2(out1)))
        out1 = self.conv2_relu(self.bn2(self.conv2(out1)))
        # print(f"DownBlock, conv2_relu: {out1.shape}")

        # right
        # out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.conv3_relu(self.bn3(self.conv3(x)))
        # print(f"DownBlock, conv3_relu: {out2.shape}")

        out2 = self.bn4(self.conv4(out2))
        # print(f"DownBlock, bn4: {out2.shape}")

        # out2 = F.relu(self.bn5(self.conv5(out2)))
        out2 = self.conv5_relu(self.bn5(self.conv5(out2)))
        # print(f"DownBlock, conv5_relu: {out2.shape}")

        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


# class ShuffleNetV2(nn.Module):
class ShuffleNetV2(PruningModule):
    def __init__(self, builder, net_size, num_classes=10):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        # self.conv0 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        # self.conv0 = nn.Conv2d(3, 24, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zero')
        self.conv0 = builder.conv_mobileNet(3, 24, kernel_size=1, groups=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(24)
        self.conv0_relu = builder.activation(width=24) # new add

        self.in_channels = 24
        self.layer1 = self._make_layer(builder, out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(builder, out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(builder, out_channels[2], num_blocks[2])

        # self.conv2 = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv56 = builder.conv_mobileNet(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.conv56_relu = builder.activation(width=out_channels[3]) # new add

        self.fc3 = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, builder, out_channels, num_blocks):
        layers = [DownBlock(builder, self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(builder, out_channels, is_last=(i == num_blocks - 1)))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"x.shape: {x.shape}")

        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv0_relu(self.bn0(self.conv0(x)))
        # print(f"conv0_relu: {out.shape}")

        out, _ = self.layer1(out)
        # print(f"layer 1: {out.shape}")

        out, _ = self.layer2(out)
        # print(f"layer 2: {out.shape}")

        out, _ = self.layer3(out)
        # print(f"layer 3: {out.shape}")

        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv56_relu(self.bn2(self.conv56(out)))
        # print(f"conv56_relu: {out.shape}")

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc3(out)
        return out


configs = {
    0.2: {
        'out_channels': (40, 80, 160, 512),
        'num_blocks': (3, 3, 3)
    },

    0.3: {
        'out_channels': (40, 80, 160, 512),
        'num_blocks': (3, 7, 3)
    },

    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


# def ShuffleV2(**kwargs):
#     model = ShuffleNetV2(net_size=1, **kwargs)
#     return model


def build_shufflenetV2(version, num_classes, mask=False, batch_size=128):
    # config is useless
    assert version == "shufflenetV2"
    print("Version: {}".format(version))
    print("Num classes: {}".format(num_classes))
    print("Mask: {}".format(mask))
    print("Batch size for relu: {}\n".format(batch_size))

    builder = ModelBuilder(mask=mask, batch_size=batch_size)
    model = ShuffleNetV2(builder, net_size=1, num_classes=num_classes)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable_parameters == 1263278 # fro cifar10 

    return model


if __name__ == '__main__':
    import time

    model_version = "shufflenetV2"
    model = build_shufflenetV2(model_version, num_classes=10, mask=False, batch_size=128)
    # model = ShuffleV2(num_classes=10)

    x = torch.randn(1, 3, 32, 32)
    logit = model(x)
    print(logit.shape)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) # 1263278 for cifar10
    print(num_trainable_parameters)


# relu: orginal replace is False, now is True