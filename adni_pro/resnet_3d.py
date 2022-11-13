import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import Size
from torch.autograd import Variable
from config_multi import get_args_multi



cam_w = 100
cam_sigma = 0.4
__all__ = [
    "ResNet",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    # if isinstance(out.data, torch.cuda.FloatTensor):
    #     zero_pads = zero_pads.cuda()

    out = torch.cat([out, zero_pads.cuda()], dim=1)

    return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 现在是单模态预训练和多模态一起的模型
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        # sample_size,
        # sample_duration,
        shortcut_type="B",
        num_classes=get_args_multi().class_num, #分类数
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv3d(
                1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        #self.layer4 = self._make_layer(block, 256, layers[3], shortcut_type, stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        #self.classify = nn.Linear(512 * block.expansion, num_classes, bias=False)
        #self.classify = nn.Linear(256 * block.expansion, num_classes, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def get_cam(self, x_init_shape, fmap):
    #     x_init_shape = x_init_shape[2:]
    #     _fn = nn.Upsample(size=x_init_shape, mode="trilinear", align_corners=True)
    #     weight = self.classify.weight
    #     cams = _fn(
    #         F.conv3d(
    #             fmap, weight.detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None
    #         )
    #     )
    #     return cams

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        out = self.avgpool(feature)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        #cls = self.classify(out)
        return out

    # def twoRes
    #     self.res1 = resnet10(
    #
    #     )
    # self.res2
    # self.classify

def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def get_net(net_name):
    if "10" in net_name:
        return resnet10()
    elif "18" in net_name:
        return resnet18()
    elif "34" in net_name:
        return resnet34()
    elif "50" in net_name:
        return resnet50()
    elif "101" in net_name:
        return resnet101()

# x = torch.randn(1, 1, 100, 100, 100).cuda()
# net = get_net('ResNet18').cuda()
# print(net)
# y = net(x)
# print('size of y:', y.size())

# x = torch.randn(1, 3, 150, 180, 150)
# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
# print(model)
# y = model(x)
# print('size of y:', y.size())