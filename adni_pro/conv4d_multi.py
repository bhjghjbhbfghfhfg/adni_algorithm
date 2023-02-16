import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
import math
from config_multi import get_args_multi




class convNd(nn.Module):
    """Some Information about convNd"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size: Tuple,
                 stride,
                 padding,
                 is_transposed=False,
                 padding_mode='zeros',
                 output_padding=0,
                 dilation: int = 1,
                 groups: int = 1,
                 rank: int = 0,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(convNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, Tuple):
            padding = tuple(padding for _ in range(num_dims))
        if not isinstance(output_padding, Tuple):
            output_padding = tuple(output_padding for _ in range(num_dims))
        if not isinstance(dilation, Tuple):
            dilation = tuple(dilation for _ in range(num_dims))

        # This parameter defines which Pytorch convolution to use as a base, for 3 Conv2D is used
        if rank == 0 and num_dims <= 3:
            max_dims = num_dims - 1
        else:
            max_dims = 3

        if is_transposed:
            self.conv_f = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[max_dims - 1]
        else:
            self.conv_f = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[max_dims - 1]

        assert len(kernel_size) == num_dims, \
            'nD kernel size expected!'
        assert len(stride) == num_dims, \
            'nD stride size expected!'
        assert len(padding) == num_dims, \
            'nD padding size expected!'
        assert len(output_padding) == num_dims, \
            'nD output_padding size expected!'
        assert sum(dilation) == num_dims, \
            'Dilation rate other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.rank = rank
        self.is_transposed = is_transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.bias_initializer is not None:
            if self.use_bias:
                self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0]

        for _ in range(next_dim_len):
            if self.num_dims - 1 > max_dims:
                # Initialize a Conv_n-1_D layer
                conv_layer = convNd(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    use_bias=self.use_bias,
                                    num_dims=self.num_dims - 1,
                                    rank=self.rank - 1,
                                    is_transposed=self.is_transposed,
                                    kernel_size=self.kernel_size[1:],
                                    stride=self.stride[1:],
                                    groups=self.groups,
                                    dilation=self.dilation[1:],
                                    padding=self.padding[1:],
                                    padding_mode=self.padding_mode,
                                    output_padding=self.output_padding[1:],
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

            else:
                # Initialize a Conv layer
                # bias should only be applied by the top most layer, so we disable bias in the internal convs
                conv_layer = self.conv_f(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         bias=False,
                                         kernel_size=self.kernel_size[1:],
                                         dilation=self.dilation[1:],
                                         stride=self.stride[1:],
                                         padding=self.padding[1:],
                                         padding_mode=self.padding_mode,
                                         groups=self.groups)
                if self.is_transposed:
                    conv_layer.output_padding = self.output_padding[1:]

                # Apply initializer functions to weight and bias tensor
                if self.kernel_initializer is not None:
                    self.kernel_initializer(conv_layer.weight)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Pad the input if is not transposed convolution
        if not self.is_transposed:
            padding = list(self.padding)
            # Pad input if this is the parent convolution ie rank=0
            if self.rank == 0:
                inputShape = list(input.shape)
                inputShape[2] += 2 * self.padding[0]
                padSize = (0, 0, self.padding[0], self.padding[0])
                padding[0] = 0
                if self.padding_mode is 'zeros':
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize, 'constant',
                                  0).view(inputShape)
                else:
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize,
                                  self.padding_mode).view(inputShape)

        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        if not self.is_transposed:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [math.floor((size_i[x] + 2 * padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
            # Compute size of the output without stride
            size_ons = tuple([size_i[x] - size_k[x] + 1 for x in range(len(size_i))])
        else:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [(size_i[x] - 1) * self.stride[x] - 2 * self.padding[x] + (size_k[x] - 1) + 1 + self.output_padding[x]
                 for x in range(len(size_i))])

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b, self.out_channels) + size_o[1:], device=input.device)]
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate inputs first dimmension
            for j in range(size_i[0]):

                # Add results to this output frame
                if self.is_transposed:
                    out_frame = i + j * self.stride[0] - self.padding[0]
                else:
                    out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_ons[0]) // 2 - (1 - size_k[0] % 2)
                    k_center_position = out_frame % self.stride[0]
                    out_frame = math.floor(out_frame / self.stride[0])
                    if k_center_position != 0:
                        continue

                if out_frame < 0 or out_frame >= size_o[0]:
                    continue

                # Prepate input for next dimmension
                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = \
                    self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b, resultShape[1], -1)
            for k in range(self.out_channels):
                result[:, k, :] += self.bias[k]
            return result.view(resultShape)
        else:
            return result


class _BatchNorm4d(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(channel_size)

    def forward(self, x):
        shape_x = x.shape
        out = self.norm(x.view(shape_x[0], shape_x[1], shape_x[2] * shape_x[3] * shape_x[4] * shape_x[5])).view(shape_x)
        return out

class Conv4d_(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2,
           stride: int = 1, padding: int = 0, padding_mode: str = "zeros",
           bias: bool = True, groups: int = 1, dilation: int = 1):
        super().__init__()
        w = torch.rand(1)[0]
        if bias:
            b = torch.zeros(1)[0]
        self.kernel_size = kernel_size
        if kernel_size==1:
            self.conv = convNd(in_channels=in_channels, out_channels=out_channels,
             num_dims=4, kernel_size=kernel_size,
             stride=stride, padding=padding,
             padding_mode=padding_mode, output_padding=0,
             is_transposed=False, use_bias=bias, groups=groups, dilation=dilation,
             kernel_initializer=lambda x: torch.nn.init.constant_(x, w),
             bias_initializer=lambda x: torch.nn.init.constant_(x, b))
        else:
            self.conv = convNd(in_channels=in_channels, out_channels=out_channels*2,
                               num_dims=4, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               padding_mode=padding_mode, output_padding=0,
                               is_transposed=False, use_bias=bias, groups=groups, dilation=dilation,
                               kernel_initializer=lambda x: torch.nn.init.constant_(x, w),
                               bias_initializer=lambda x: torch.nn.init.constant_(x, b))
        self.out_channels = out_channels

    def forward(self, x):
        if self.kernel_size == 1:
            return self.conv(x)
        out = self.conv(x)
        n_batch, _, _, n_fig1, n_fig2, n_fig3 = out.shape
        return out.view(n_batch, self.out_channels, 2, n_fig1, n_fig2, n_fig3)

class Conv4d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2,
           stride: int = 1, padding: int = 0, padding_mode: str = "zeros",
           bias: bool = True, groups: int = 1, dilation: int = 1):
        super(Conv4d, self).__init__()

        if isinstance(stride, tuple):
            stride = stride[1:]
        else:
            stride = stride

        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[1:]
        else:
            kernel_size = kernel_size

        if isinstance(padding, tuple):
            padding = padding[1:]
        else:
            padding = padding

        self.conv_mri = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)
        self.conv_pet = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)

        # torch.empty()用来返回一个没有初始化的 tensor
        # nn.Parameter() 可以让一个 普通的tensor ==>> 变量不断学习的tensor
        self.w_mri = nn.Parameter(torch.empty((out_channels, out_channels)))
        # 使用 凯明正态分布 初始化卷层参数，a是默认值
        nn.init.kaiming_uniform_(self.w_mri, a=math.sqrt(5))
        self.w_pet = nn.Parameter(torch.empty((out_channels, out_channels)))
        nn.init.kaiming_uniform_(self.w_pet, a=math.sqrt(5))

        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        # 没理解？？？
        x_mri = x[:, :, 0, :, :, :]
        x_pet = x[:, :, 1, :, :, :]

        # 获取卷积之后的feature
        h_mri = self.conv_mri(x_mri)
        h_pet = self.conv_pet(x_pet)

        # mri attend to pet
        # b,1,a,b,c
        b, c, x, y, z = h_mri.shape
        # 获取从mri 对 pet的每个部分关注权重
        score_pet = torch.einsum('bcxyz,cd,bdxyz->bxyz', h_mri, self.w_mri, h_pet)#/self.out_channels**0.5
        # 将获取的权重映射在 【0， 1】之间
        attn_pet = score_pet.view(b, -1).softmax(-1).view(b, 1, x, y, z)
        # 将权重和pet feature乘机获取 value
        context_pet = attn_pet * h_pet


        # pet attend to mri
        score_mri = torch.einsum('bcxyz,cd,bdxyz->bxyz', h_pet, self.w_pet, h_mri)#/self.out_channels ** 0.5
        # view()可以理解为展示形式，比如 x * x矩阵
        attn_mri = score_mri.view(b, -1).softmax(-1).view(b, 1, x, y, z)
        context_mri = attn_mri * h_mri

        # 将获取的Attention value 和 原MRI feature map 加和，防止学习到的知识全是别人的。
        ret_mri = h_mri + context_pet
        ret_pet = h_pet + context_mri
        # 把 ret_mri 和 ret_pet 从新压缩成4d的
        return torch.stack([ret_mri, ret_pet], 2)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return Conv4d(
        in_planes,
        out_planes,
        kernel_size=(2, 3, 3, 3),
        stride=stride,
        padding=(0,1,1,1),
        bias=False,
    )




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

class BatchNorm4by3(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.norm = torch.nn.BatchNorm3d(channel_size)

    def forward(self, x):
        n_batch = x.size(0)
        n_channel = x.size(1)
        n_figs = x.size(2)
        n_fig1 = x.size(3)
        n_fig2 = x.size(4)
        n_fig3 = x.size(5)

        x_3d = x.view(n_batch, n_channel, n_figs*n_fig1, n_fig2, n_fig3)
        out = self.norm(x_3d)
        out = out.view(n_batch, n_channel, n_figs, n_fig1, n_fig2, n_fig3)
        return out

class BatchNorm4by3S(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm3d(channel_size)
        self.norm2 = torch.nn.BatchNorm3d(channel_size)

    def forward(self, x):
        x1 = x[:,:,0]
        x2 = x[:,:,1]
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        return torch.stack([x1, x2], 2)

class _MaxPool4d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        bsz, inch, ha, wa, hb, wb = x.size()
        out1 = x.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.maxpool(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = out1.size()
        out2 = out1.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.maxpool(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        return out2


class MaxPool4by3(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPool4by3, self).__init__()
        self.pool_3d = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        n_batch = x.size(0)
        n_channel = x.size(1)
        n_figs = x.size(2)
        n_fig1 = x.size(3)
        n_fig2 = x.size(4)
        n_fig3 = x.size(5)

        x_3d = x.view(n_batch, n_figs*n_channel, n_fig1, n_fig2, n_fig3)
        out = self.pool_3d(x_3d)
        out = out.view(n_batch, n_channel, n_figs, out.size(-3), out.size(-2), out.size(-1))
        return out


class AdaptiveAVGPool4by3(nn.Module):
    def __init__(self, size):
        super(AdaptiveAVGPool4by3, self).__init__()
        self.adaptive_pool_3d = nn.AdaptiveAvgPool3d(size)
        self.size = size

    def forward(self, x):
        n_batch = x.size(0)
        n_channel = x.size(1)
        n_figs = x.size(2)
        n_fig1 = x.size(3)
        n_fig2 = x.size(4)
        n_fig3 = x.size(5)


        x_3d = x.view(n_batch, n_figs*n_channel, n_fig1, n_fig2, n_fig3)
        out = self.adaptive_pool_3d(x_3d).view(n_batch, n_channel, n_figs, *self.size)
        out = out.mean(2)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm4by3S(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = BatchNorm4by3S(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out

class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            # sample_size,
            # sample_duration,
            shortcut_type="B",
            num_classes=get_args_multi().class_num,  # 分类数
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.layer0 = nn.Sequential(
            Conv4d(1, 64, kernel_size=(2,7,7,7), stride=(1,1,2,2), padding=(0,3,3,3), bias=False),
            BatchNorm4by3(64),
            nn.ReLU(inplace=True),
            MaxPool4by3(kernel_size=(3, 3, 3), stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1)

        self.avgpool = AdaptiveAVGPool4by3((1, 1, 1))
        self.classify = nn.Linear(512 * block.expansion, num_classes, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    Conv4d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    BatchNorm4by3S(planes * block.expansion),
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

    def forward(self, x1, x2):
        x = torch.stack([x1,x2],dim=2)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        out = self.avgpool(feature)
        out = out.view(out.size(0), -1)
        cls = self.classify(out)
        return cls

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