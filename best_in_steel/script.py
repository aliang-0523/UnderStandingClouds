#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/29 上午9:10
# software: PyCharm
import abc
import yaml
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import pandas as pd
import os
import numpy as np
import re
import math
import collections
from functools import partial
from copy import deepcopy
import logging.config
import datetime
import cv2
import warnings
from albumentations import HorizontalFlip, Normalize, Compose, ImageOnlyTransform
from albumentations.pytorch import ToTensor
from collections import Sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.jit import load
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torch.utils import model_zoo
from pathlib import Path
from os.path import join
from collections import OrderedDict
from scipy import ndimage
__all__ = ['SENet','senet154', 'se_resnet50', 'se_resnet101','se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']
pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out
class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def cycle_rgb_weights(weights, n):
    """Repeat RGB weights n times. Assumes channels are dim 1"""
    slices = [(c % 3, c % 3 + 1) for c in range(n)]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights
def select_rgb_weights(weights, rgb_str):
    """Repeat RGB weights given a str eg. RRGGBB would repeat each weight twice"""
    rgb_str = rgb_str.lower()
    rgb_map = {'r': 0, 'g': 1, 'b': 2}
    slices = [(rgb_map[c] % 3, rgb_map[c] % 3 + 1) for c in rgb_str]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights
def avg_rgb_weights(weights):
    new_weights = weights.mean(dim=1, keepdim=True)
    return new_weights
def transfer_weights(pretrained, replacement, method='cycle'):
    """
    Transform pretrained weights to be used for a layer with a different number of channels.
    """
    if method == 'cycle':
        n = replacement.in_channels
        weights = cycle_rgb_weights(pretrained.weight, n)
    elif method == 'avg':
        weights = avg_rgb_weights(pretrained.weight)
    else:
        raise NotImplementedError('`method` must be "cycle" or "avg", received {}'.format(method))
    replacement.weight = nn.Parameter(weights)
    return replacement
class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x
class EfficientNetEncoder(nn.Module):
    """
    Implementation taken from: https://github.com/lukemelas/EfficientNet-PyTorch
    """

    def __init__(self,
        width_coeff,
        depth_coeff,
        image_size,
        dropout_rate,
        drop_connect_rate,
        block_chunks,
        in_channels = 3  # rgb
    ):
        super().__init__()
        self._blocks_args = self.get_block_args()
        self._global_params = get_global_params(width_coeff, depth_coeff, image_size,
                                                dropout_rate, drop_connect_rate)
        self.block_chunks = block_chunks
        self.in_channels = in_channels

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(self.in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        n_blocks = len(self._blocks)
        assert self.block_chunks[-1] == n_blocks, f'{self.block_chunks[-1]}, {n_blocks}'
        self._out_shapes = [self._bn0.num_features]
        self._out_shapes += [self._blocks[i - 1]._bn2.num_features for i in self.block_chunks[1:]]
        self._out_shapes = list(reversed(self._out_shapes))
        # precalc drop connect rates
        self.drop_connect_rates = np.arange(0, n_blocks, dtype=np.float) / n_blocks
        self.drop_connect_rates *= self._global_params.drop_connect_rate

    def forward_blocks(self, x, start_idx, end_idx):
        for idx in range(start_idx, end_idx):
            x = self._blocks[idx](x, self.drop_connect_rates[idx])
        return x

    def forward(self, x):
        x0 = relu_fn(self._bn0(self._conv_stem(x)))
        x1 = self.forward_blocks(x0, self.block_chunks[0], self.block_chunks[1])
        x2 = self.forward_blocks(x1, self.block_chunks[1], self.block_chunks[2])
        x3 = self.forward_blocks(x2, self.block_chunks[2], self.block_chunks[3])
        x4 = self.forward_blocks(x3, self.block_chunks[3], self.block_chunks[4])
        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('_conv_head.weight')
        state_dict.pop('_bn1.weight')
        state_dict.pop('_bn1.bias')
        state_dict.pop('_bn1.running_mean')
        state_dict.pop('_bn1.running_var')
        state_dict.pop('_bn1.num_batches_tracked')
        state_dict.pop('_fc.bias')
        state_dict.pop('_fc.weight')

        if self.in_channels != 3:
            state_dict = self.modify_in_channel_weights(state_dict, self.in_channels)

        super().load_state_dict(state_dict, **kwargs)

    def modify_in_channel_weights(self, state_dict, in_channels):
        pretrained = state_dict['_conv_stem.weight']
        cycled_weights = cycle_rgb_weights(pretrained, in_channels)
        state_dict['_conv_stem.weight'] = cycled_weights
        return state_dict

    def get_block_args(self):
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25',
            'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]
        blocks_args = BlockDecoder.decode(blocks_args)
        return blocks_args

    def info(self):
        msg = '== EfficientNetEncoder ==\n\n'
        msg += 'x0: conv_stem\n'
        chunks = zip(self.block_chunks[:-1], self.block_chunks[1:])
        for i, (start, end) in enumerate(chunks):
            msg += f'x{i+1}: blocks[{start}:{end}]\n'
        msg += '\n'
        msg += f'Out shapes (x4, x3, x2, x1, x0): {self._out_shapes}\n'
        msg += str(self._global_params)
        print(msg)
url_map = {
    'efficientnet-b0': '../input/efficientnet-pytorch/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': '../input/efficientnet-pytorch/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': '../input/efficientnet-pytorch/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': '../input/efficientnet-pytorch/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': '../input/efficientnet-pytorch/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': '../input/efficientnet-pytorch/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': '../input/efficientnet-pytorch/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': '../input/efficientnet-pytorch/efficientnet-b7-dcc49843.pth',
}

efficientnet_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b0'],
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (320, 112, 40, 24, 32),
        'params': {
            'width_coeff': 1.0,
            'depth_coeff': 1.0,
            'image_size': 224,
            'dropout_rate': 0.2,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 2, 5, 11, 16]
        },
    },

    'efficientnet-b1': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b1'],
                'input_space': 'RGB',
                'input_size': [3, 240, 240],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (320, 112, 40, 24, 32),
        'params': {
            'width_coeff': 1.0,
            'depth_coeff': 1.1,
            'image_size': 240,
            'dropout_rate': 0.2,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 3, 8, 16, 23]
        },
    },

    'efficientnet-b2': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b2'],
                'input_space': 'RGB',
                'input_size': [3, 260, 260],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (352, 120, 48, 24, 32),
        'params': {
            'width_coeff': 1.1,
            'depth_coeff': 1.2,
            'image_size': 260,
            'dropout_rate': 0.3,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 3, 8, 16, 23]
        },
    },

    'efficientnet-b3': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b3'],
                'input_space': 'RGB',
                'input_size': [3, 300, 300],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (384, 136, 48, 32, 40),
        'params': {
            'width_coeff': 1.2,
            'depth_coeff': 1.4,
            'image_size': 300,
            'dropout_rate': 0.3,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 3, 8, 18, 26]
        },
    },

    'efficientnet-b4': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b4'],
                'input_space': 'RGB',
                'input_size': [3, 380, 380],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (448, 160, 56, 32, 48),
        'params': {
            'width_coeff': 1.4,
            'depth_coeff': 1.8,
            'image_size': 380,
            'dropout_rate': 0.4,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 5, 10, 22, 32]
        },
    },

    'efficientnet-b5': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b5'],
                'input_space': 'RGB',
                'input_size': [3, 456, 456],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (512, 176, 64, 40, 48),
        'params': {
            'width_coeff': 1.6,
            'depth_coeff': 2.2,
            'image_size': 456,
            'dropout_rate': 0.4,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 6, 13, 27, 39]
        },
    },

    'efficientnet-b6': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b6'],
                'input_space': 'RGB',
                'input_size': [3, 528, 528],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (576, 200, 72, 40, 56),
        'params': {
            'width_coeff': 1.8,
            'depth_coeff': 2.6,
            'image_size': 528,
            'dropout_rate': 0.5,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 7, 15, 31, 45]
        },
    },

    'efficientnet-b7': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b7'],
                'input_space': 'RGB',
                'input_size': [3, 600, 600],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (640, 224, 80, 48, 64),
        'params': {
            'width_coeff': 2.0,
            'depth_coeff': 3.1,
            'image_size': 600,
            'dropout_rate': 0.5,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 9, 18, 38, 55]
        },
    },
}
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])
# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])
# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)
def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)
class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def get_global_params(width_coeff, depth_coeff, image_size, dropout_rate, drop_connect_rate):
    """ Map EfficientNet model name to parameter coefficients. """

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=1000,
        width_coefficient=width_coeff,
        depth_coefficient=depth_coeff,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )
    return global_params
class swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(torch.nn.Module):
    """
    https://github.com/qubvel/segmentation_models.pytorch/pull/85/files
    """
    def forward(self, x):
        return swish.apply(x)
class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
class SENetEncoder(SENet):

    def __init__(self, in_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        self.in_channels = in_channels

        del self.last_linear
        del self.avg_pool
        self.layer0[0] = nn.Conv2d(in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)

    def forward(self, x):
        for module in self.layer0[:-1]:
            x = module(x)

        x0 = x
        x = self.layer0[-1](x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')

        if self.in_channels != 3:
            state_dict = self.modify_in_channel_weights(state_dict, self.in_channels)

        super().load_state_dict(state_dict, **kwargs)

    def modify_in_channel_weights(self, state_dict, in_channels):
        self.layer0[0] = nn.Conv2d(in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)
        pretrained = state_dict['layer0.conv1.weight']
        cycled_weights = cycle_rgb_weights(pretrained, in_channels)
        state_dict['layer0.conv1.weight'] = cycled_weights
        return state_dict
senet_encoders = {
    'senet154': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['senet154'],
        'out_shapes': (2048, 1024, 512, 256, 128),
        'params': {
            'block': SEBottleneck,
            'dropout_p': 0.2,
            'groups': 64,
            'layers': [3, 8, 36, 3],
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnet50': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNetBottleneck,
            'layers': [3, 4, 6, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 1,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnet101': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNetBottleneck,
            'layers': [3, 4, 23, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 1,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnet152': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNetBottleneck,
            'layers': [3, 8, 36, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 1,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnext50_32x4d': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnext50_32x4d'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNeXtBottleneck,
            'layers': [3, 4, 6, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 32,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnext101_32x4d': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnext101_32x4d'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNeXtBottleneck,
            'layers': [3, 4, 23, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 32,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },
}
encoders = {}
encoders.update(efficientnet_encoders)
encoders.update(senet_encoders)

def get_encoder(name, encoder_weights=None, in_channels=3):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(in_channels=in_channels, **encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        r = encoder.load_state_dict(torch.load(settings['url']))
        print(f'Load result: {r}')
    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained='imagenet'):
    settings = encoders[encoder_name]['pretrained_settings']

    if pretrained not in settings.keys():
        raise ValueError('Avaliable pretrained options {}'.format(settings.keys()))

    formatted_settings = {}
    formatted_settings['input_space'] = settings[pretrained].get('input_space')
    formatted_settings['input_range'] = settings[pretrained].get('input_range')
    formatted_settings['mean'] = settings[pretrained].get('mean')
    formatted_settings['std'] = settings[pretrained].get('std')
    return formatted_settings

'''
def get_preprocessing_fn(encoder_name, pretrained='imagenet'):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return partial(preprocess_input, **params)
'''
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x

        if skip is not None:
            skipsize = (skip.shape[-2], skip.shape[-1])
            x = F.interpolate(x, size=skipsize, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.block(x)
        x = self.attention2(x)
        return x
class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            attention_type=None
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, weight_std=False, upsample=False, activation='relu'):

        if weight_std:
            Conv2d = Conv2dWS
        else:
            Conv2d = nn.Conv2d

        if activation == 'relu':
            relu_fn = nn.ReLU(inplace=True)
        elif activation == 'swish':
            relu_fn = Swish()
        else:
            raise ValueError(f'`activation` must be "relu" or "swish"')

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            relu_fn,
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, weight_std, n_upsamples=0, activation='relu'):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples), activation=activation)
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(
                    out_channels, out_channels, weight_std, upsample=True, activation=activation))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPNDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_upsampling=4,
            final_channels=1,
            dropout=0.2,
            weight_std=False,
            merge_policy='add',
            activation='relu'
    ):
        super().__init__()

        if merge_policy not in ['add', 'cat']:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(merge_policy))
        self.merge_policy = merge_policy

        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 3, activation)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 2, activation)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 1, activation)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 0, activation)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        if self.merge_policy == 'cat':
            segmentation_channels *= 4

        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        if self.merge_policy == 'add':
            x = s5 + s4 + s3 + s2
        elif self.merge_policy == 'cat':
            x = torch.cat([s5, s4, s3, s2], dim=1)

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=True)
        return x


# -- Model

class Unet(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
        attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            attention_type=None,
            in_channels=3
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=attention_type
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)


class FPN(EncoderDecoder):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        dropout: spatial dropout rate in range (0, 1).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_activation='relu',
            classes=1,
            dropout=0.2,
            activation='sigmoid',
            in_channels=3,
            weight_std=False,
            final_upsampling=4,
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )

        decoder = FPNDecoder(
            encoder_channels=encoder.out_shapes,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            final_channels=classes,
            dropout=dropout,
            weight_std=weight_std,
            final_upsampling=4,
            merge_policy=decoder_merge_policy,
            activation=decoder_activation
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'fpn-{}'.format(encoder_name)


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = torch.load(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b' + str(i) for i in range(num_models)]
        if model_name.replace('-', '_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class EffNet(nn.Module):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """

    def __init__(self, encoder_name, classes, pretrained, in_channels='rgb', verbose=0):
        super().__init__()
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(encoder_name, num_classes=classes)
        else:
            self.encoder = EfficientNet.from_name(
                encoder_name,
                override_params={'num_classes': classes})

        # modify input channels
        Conv2d = get_same_padding_conv2d(image_size=self.encoder._global_params.image_size)
        conv_stem = Conv2d(
            len(in_channels),
            self.encoder._conv_stem.out_channels,
            kernel_size=3,
            stride=2,
            bias=False
        )
        self.transfer_weights(self.encoder._conv_stem, conv_stem, in_channels)
        self.encoder._conv_stem = conv_stem

        self.decoder = Noop()  # for compatibility with segmentation models

    def forward(self, x):
        return self.encoder(x)

    def __str__(self):
        return str(self.encoder)

    def __repr__(self):
        return self.__str__()

    def transfer_weights(self, pretrained_layer, replacement_layer, rgb_str='rgb'):
        """
        Transform pretrained weights to be used for a layer with a different number of channels.
        """
        weights = select_rgb_weights(pretrained_layer.weight, rgb_str)
        replacement_layer.weight = nn.Parameter(weights)
        return replacement_layer


class Noop(nn.Module):

    def forward(self, x):
        return x


class ChannelTranspose(ImageOnlyTransform):
    def get_transform_init_args_names(self):
        return ()

    def get_params_dependent_on_targets(self, params):
        pass

    def __init__(self, axes=(2, 0, 1)):
        super().__init__(always_apply=True)
        self.axes = axes

    def apply(self, img, **params):
        return np.transpose(img, self.axes)


class AugmentationBase(abc.ABC):
    MEAN = [0.3439]
    STD = [0.0383]

    H = 256
    W = 1600

    def __init__(self):
        self.transform = self.notimplemented

    def build_transforms(self, train):
        if train:
            self.transform = self.build_train()
        else:
            self.transform = self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    def build_test(self):
        return Compose([
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])

    def notimplemented(self, *args, **kwargs):
        raise Exception('You must call `build_transforms()` before using me!')

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def copy(self):
        return deepcopy(self)


class LightTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class CompatabilityTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])

    def build_test(self):
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ChannelTranspose()
        ])


# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/augmentation.py -- END --
# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/datasets.py -- START --

class SteelDataset(Dataset):
    img_folder = 'implement me!'
    N_CLASSES = 4
    rle_cols = [f'rle{i}' for i in range(N_CLASSES)]

    def __init__(self, df, data_dir, transforms):
        self.df = df
        self.data_dir = data_dir / self.img_folder
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def read_greyscale(self, idx):
        f = self.fnames[idx]
        return f, cv2.imread(str(self.data_dir / f))[:, :, 0:1]  # select one channel

    def rle(self, idx):
        return self.df.iloc[idx][self.rle_cols]

    def __len__(self):
        return len(self.fnames)


class SteelDatasetTest(SteelDataset):
    img_folder = 'test_images'

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms.build_transforms(train=False)

    def __getitem__(self, idx):
        f, image = self.read_greyscale(idx)
        images = self.transforms(image=image)["image"]
        return f, images
class SteelDatasetTrain(SteelDataset):
    img_folder = 'train_images'

    def __init__(self, df, data_dir, transforms,train=True):
        super().__init__(df, data_dir, transforms)
        self.transforms.build_transforms(train=train)

    def __getitem__(self, idx):
        f, image = self.read_greyscale(idx)
        images = self.transforms(image=image)["image"]
        return f, images

# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/datasets.py -- END --
# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/data_loaders.py -- START --

class SteelClasTestDataLoader(DataLoader):
    test_csv = '../sample_submission.csv'

    def __init__(self, transforms, data_dir, batch_size, nworkers, pin_memory=True):
        self.transforms = transforms
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.test_df = self.load_df()

        dataset = SteelDatasetTest(self.test_df, self.data_dir, transforms.copy())
        super().__init__(dataset, batch_size=self.bs, num_workers=nworkers, pin_memory=pin_memory)

    def load_df(self):
        df = pd.read_csv(self.test_csv)
        df['ImageId'], df['ClassId'] = zip(*df['Image_Label'].str.split('_'))
        df['ClassId'] = df['ClassId'].map({'Fish':0,'Flower':1,'Gravel':2,'Sugar':3})
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df.columns = [f'rle{c}' for c in range(4)]
        return df

    def split_validation(self):
        raise Exception('Attempt to split a validation data_loader from a test-only data_loader')


class SteelDataLoader(DataLoader):
    train_csv = 'train.csv'
    test_csv = '../sample_submission.csv'

    def __init__(self, transforms, data_dir, batch_size, shuffle,
                 validation_split, nworkers, pin_memory=True, train=True, alpha=None, balance=None
                 ):  # noqa
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(train, validation_split)

        if train:
            dataset = SteelDatasetTrain(self.train_df, self.data_dir, transforms.copy(), True)
        else:
            dataset = SteelDatasetTest(self.train_df, self.data_dir, transforms.copy())

        if train and balance is not None and alpha is not None:
            class_idxs = self.sort_classes(self.train_df)
            n_batches = self.train_df.shape[0] // batch_size
            sampler = SamplerFactory(2).get(class_idxs, batch_size, n_batches, alpha, balance)
            super().__init__(dataset, batch_sampler=sampler,
                             num_workers=nworkers, pin_memory=pin_memory)
        else:
            super().__init__(dataset, batch_size, shuffle=shuffle,
                             num_workers=nworkers, pin_memory=pin_memory)

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.test_csv
        df = pd.read_csv(csv_filename)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df.columns = [f'rle{c}' for c in range(4)]
        df['defects'] = df.count(axis=1)

        # add classification columns
        for c in range(4):
            df[f'c{c}'] = df[f'rle{c}'].apply(lambda rle: not pd.isnull(rle))

        if train and validation_split > 0:
            return train_test_split(df, test_size=validation_split, stratify=df["defects"])

        return df, pd.DataFrame({})

    def sort_classes(self, df):
        counts = {c: df[f'c{c}'].sum() for c in range(4)}
        sorted_classes = sorted(counts.items(), key=lambda kv: kv[1])

        def assign_min_sample_class(row, sorted_classes):
            for c, _ in sorted_classes:
                if row[f'c{c}']:
                    return c
            return -1

        df['sample_class'] = df.apply(
            lambda row: assign_min_sample_class(row, sorted_classes), axis=1)
        class_idxs = [list(np.where(df['sample_class'] == c)[0]) for c in range(-1, 4)]
        return class_idxs

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            dataset = SteelDatasetTrain(
                self.val_df, self.data_dir, self.transforms.copy(), False)
            return DataLoader(dataset, self.bs // 2,
                              num_workers=self.nworkers, pin_memory=self.pin_memory)


# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/data_loaders.py -- END --
# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/preprocess.py -- START --

class RLE:
    """
    Encapsulates run-length-encoding functionality.
    """

    MASK_H = 256
    MASK_W = 1600

    @classmethod
    def from_str(cls, s):
        if s != s:
            return cls()
        list_ = [int(n) for n in s.split(' ')]
        return cls.from_list(list_)

    @classmethod
    def from_mask(cls, mask):
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return cls.from_list(runs)

    @classmethod
    def from_list(cls, list_):
        n_items = int(len(list_) / 2)
        items = np.zeros((n_items, 2), dtype=np.uint64)
        for i in range(n_items):
            items[i, 0] = list_[i * 2]
            items[i, 1] = list_[i * 2 + 1]
        return cls(items)

    def __init__(self, items=np.zeros((0, 0))):
        self._items = items

    @property
    def items(self):
        return self._items

    def __iter__(self):
        for idx, item in enumerate(self.items):
            yield (item[0], item[1])  # run, length

    def __len__(self):
        return self.items.shape[0]

    def to_mask(self):
        mask = np.zeros(self.MASK_H * self.MASK_W, dtype=np.uint8)
        for run, length in self:
            run = int(run - 1)
            end = int(run + length)
            mask[run:end] = 1
        return mask.reshape(self.MASK_H, self.MASK_W, order='F')

    def to_str_list(self):
        list_ = []
        for run, length in self:
            list_.append(str(run))
            list_.append(str(length))
        return list_

    def __str__(self):
        if len(self) == 0:
            return ''
        return ' '.join(self.to_str_list())

    def __repr__(self):
        return self.__str__()


class PostProcessor:
    N_CLASSES = 4
    MIN_COMPONENT_SIZE = 200

    def __init__(
            self,
            classification_factor=5,
            domination=False,
            dilation=0,
            p_thresh=None,
            min_class_sizes=None
    ):
        self.clas_factor = classification_factor
        self.domination = domination
        self.dilation = dilation
        self.p_thresh = np.array(p_thresh)
        self.min_class_sizes = np.array(min_class_sizes)
        print(self.clas_factor)
        print(self.domination)
        print(self.dilation)
        print(self.p_thresh)
        print(self.min_class_sizes)
        self.n_wipes = 0

    def _component_domination(self, preds):
        """
        Ensure that no predictions in the multi-channel mask overlap. Larger predicted components
        will overwrite overlapping predictions.
        """
        components, component_sizes = self._find_components(preds)
        mask = self._write_preds(components, component_sizes)
        return mask

    def _find_components(self, preds):
        C, H, W = preds.shape
        total_components = 0
        channel_components = []
        for c in range(C):
            max_label, labelled_components = cv2.connectedComponents(preds[c].astype(np.uint8))
            n_components = max_label - 1
            labelled_components[labelled_components > 0] += total_components  # offset labels
            total_components += n_components
            channel_components.append(labelled_components)
        components = np.stack(channel_components, axis=0)
        component_sizes = [(label, (components == label).sum())
                           for label in range(1, total_components + 1)]
        component_sizes = sorted(component_sizes, key=lambda item: item[1])  # sort by size
        return components, component_sizes

    def _write_preds(self, components, component_sizes):
        C, H, W = components.shape
        mask = np.zeros((C, H, W), dtype=np.uint8)
        for label, size in component_sizes:
            component_mask_3d = components == label
            component_mask_flatten = component_mask_3d.any(axis=0)
            component_mask_expand = np.repeat(component_mask_flatten[np.newaxis, :, :], C, axis=0)

            # set the mask region to zero across all channels
            if self.domination:
                mask[component_mask_expand] = 0

            # set just the channel applicable to the mask to 1
            mask[component_mask_3d] = 1
        return mask

    def process(self, prob_seg, prob_clas):
        mask = prob_seg > self.p_thresh[:, np.newaxis, np.newaxis]
        # mask = self._component_domination(mask)
        for c in range(self.N_CLASSES):
            t0 = self.min_class_sizes[c]
            t1 = (t0 * self.clas_factor) - (t0 * (self.clas_factor - 1) * prob_clas[c])
            if mask[c, :, :].sum() < t1:
                mask[c, :, :] = 0  # wipe the predictions
                self.n_wipes += 1
            elif self.dilation > 0:
                mask[c, :, :] = ndimage.binary_dilation(
                    mask[c, :, :],
                    iterations=self.dilation
                )
        return mask


# -- /home/khornlund/bb/kaggle/aptos/aptos/data_loader/preprocess.py -- END --

# -- /home/khornlund/bb/kaggle/aptos/aptos/utils/saving.py -- START --

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def ensure_parent(path):
    parent = os.path.sep.join(path.split(os.path.sep)[:-1])
    ensure_dir(parent)
    return path


def arch_path(config):
    return ensure_dir(join(config['save_dir'], config['name']))


def arch_datetime_path(config):
    start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    return ensure_dir(join(arch_path(config), start_time))


def log_path(config):
    return ensure_dir(join(arch_path(config), 'logs'))


def trainer_paths(config):
    """Returns the paths to save checkpoints and tensorboard runs. eg.
    saved/Mnist_LeNet/<start time>/checkpoints
    saved/Mnist_LeNet/<start time>/runs
    """
    arch_datetime = arch_datetime_path(config)
    return (
        ensure_dir(join(arch_datetime, 'checkpoints')),
        ensure_dir(join(arch_datetime, 'runs'))
    )


# -- /home/khornlund/bb/kaggle/aptos/aptos/utils/saving.py -- END --

# -- /home/khornlund/bb/kaggle/aptos/aptos/utils/logger.py -- START --

LOG_CONFIG = """
version: 1
disable_existing_loggers: False
formatters:
    simple:
        format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

handlers:
    console:
        class: logging.StreamHandler
        level: DEBUG
        formatter: simple
        stream: ext://sys.stdout

root:
    level: INFO
    handlers: [console]
"""


def setup_logging(run_config, log_config=LOG_CONFIG, default_level=logging.INFO):
    """
    Setup logging configuration
    """
    config = yaml.safe_load(log_config)
    logging.config.dictConfig(config)


logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


def setup_logger(cls, verbose=0):
    logger = logging.getLogger(cls.__class__.__name__)
    if verbose not in logging_level_dict:
        raise KeyError(f'verbose option {verbose} for {cls} not valid. '
                       f'Valid options are {logging_level_dict.keys()}.')
    logger.setLevel(logging_level_dict[verbose])
    return logger


class MultiClasModel(nn.Module):
    flips = [None, 3]

    # original image sizes
    H = 256
    W = 1600

    # training cropped image sizes
    h = 128
    w = 800

    def __init__(self, models):
        super().__init__()
        self.models = models
        for name, model in self.models.items():
            print(f'Loaded: {name}')

    def forward(self, x):
        h, w, = self.h, self.w
        res = []
        for name, cfg in self.models.items():
            m = cfg['model']

            # process input in chunks of same size as training data
            out0 = m(x[:, :, :h, :w])
            out1 = m(x[:, :, :h, w:])
            out2 = m(x[:, :, h:, :w])
            out3 = m(x[:, :, h:, w:])

            # take max output
            stack = torch.stack([out0, out1, out2, out3], dim=0)
            output_max = stack.max(dim=0)[0]
            res.append(torch.sigmoid(output_max))

        res = torch.stack(res, dim=0)
        return torch.mean(res, dim=0)


class MultiSegModel(nn.Module):
    flips = [None, 3]

    def __init__(self, models):
        super().__init__()
        self.models = models
        for name, model in self.models.items():
            print(f'Loaded: {name}')

    def forward(self, x):
        res = []
        total_weight = 0
        for name, cfg in self.models.items():
            m = cfg['model']
            w = cfg['weight']
            for flip in self.flips:
                if flip is None:
                    output = m(x)
                else:
                    output = m(torch.flip(x, dims=[flip]))
                    output = torch.flip(output, dims=[flip])
                res.append(torch.sigmoid(output) * w)
                total_weight += w
        res = torch.stack(res)
        return torch.sum(res, dim=0) / total_weight


def build_instance(ctor, name, config, *args):
    return ctor(*args, **config[name]['args'])


class Runner:

    def predict(self, config):
        setup_logging(config)
        self.logger = setup_logger(self, config['testing']['verbose'])
        self._seed_everything(config['seed'])
        self.logger.info(f'Using config:\n{config}')

        tsfm = LightTransforms()

        # -- classification --
        self.logger.info('-- CLASSIFICATION --')
        self.logger.debug('Getting data_loader instance')
        data_loader = SteelClasTestDataLoader(
            tsfm,
            config['testing']['data_dir'],
            batch_size=config['testing']['batch_size'],
            nworkers=config['testing']['num_workers']
        )

        self.logger.debug('Building model architecture')
        models = {}
        for arch, cfg in config['classification']['models'].items():
            self.logger.debug(f'Building {arch}')
            model = EffNet(**cfg['args'])
            model, device = self._prepare_device(model, config['n_gpu'])
            model_checkpoint = cfg['checkpoint']
            self.logger.debug(f'Loading checkpoint {model_checkpoint}')
            checkpoint = torch.load(model_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            models[arch] = {'model': model}

        model = MultiClasModel(models)
        model.eval()

        for name, cfg in model.models.items():
            assert not cfg['model'].training, 'model is still in training mode!'

        classification_preds = {}
        self.logger.debug('Starting...')
        with torch.no_grad():
            for i, (fs, data) in enumerate(tqdm(data_loader)):
                data = data.to(device)
                output = model(data)
                batch_preds = output.detach().cpu().numpy()
                for (f, preds) in zip(fs, batch_preds):
                    classification_preds[f] = preds

        del data_loader
        del models
        del model
        torch.cuda.empty_cache()

        # -- segmentation --
        self.logger.info('-- SEGMENTATION --')
        self.logger.debug('Getting data_loader instance')
        data_loader = SteelDataLoader(
            tsfm,
            config['testing']['data_dir'],
            batch_size=config['testing']['batch_size'],
            validation_split=0.0,
            train=False,
            shuffle=False,
            nworkers=config['testing']['num_workers']
        )

        self.logger.debug('Building model architecture')
        models = {}
        for arch, cfg in config['segmentation']['models'].items():
            self.logger.debug(f'Building {arch}')
            if cfg['type'] == 'Unet':
                model = Unet(**cfg['args'])
            elif cfg['type'] == 'FPN':
                model = FPN(**cfg['args'])
            model, device = self._prepare_device(model, config['n_gpu'])
            model_checkpoint = cfg['checkpoint']
            self.logger.debug(f'Loading checkpoint {model_checkpoint}')
            checkpoint = torch.load(model_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            models[arch] = {
                'model': model,
                'weight': cfg['weight']
            }

        model = MultiSegModel(models)
        model.eval()

        for name, cfg in model.models.items():
            assert not cfg['model'].training, 'model is still in training mode!'

        predictions = []
        pp = build_instance(PostProcessor, 'postprocess', config)

        self.logger.debug('Starting...')
        with torch.no_grad():
            for i, (fs, data) in enumerate(tqdm(data_loader)):
                data = data.to(device)
                output = model(data)
                batch_preds = output.detach().cpu().numpy()
                for (f, preds) in zip(fs, batch_preds):
                    masks = pp.process(preds, classification_preds[f])
                    for class_, mask in enumerate(masks):
                        rle = str(RLE.from_mask(mask))
                        name = f + f"_{class_ + 1}"
                        predictions.append([name, rle])

        self.logger.info(f'Wiped {pp.n_wipes} times during postprocessing')
        # save predictions to submission.csv
        df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
        return df

    def _prepare_device(self, model, n_gpu_use):
        device, device_ids = self._get_device(n_gpu_use)
        model = model.to(device)
        return model, device

    def _get_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                                f"but only {n_gpu} are available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        self.logger.info(f'Using device: {device}, {list_ids}')
        return device, list_ids

    def _seed_everything(self, seed):
        self.logger.info(f'Using random seed: {seed}')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

config_str = """
name: sever
n_gpu: 1
save_dir: saved/
seed: 1234

classification:

  models:

    b5-1019_084415:
      type: EffNet
      args:
        encoder_name: efficientnet-b5
        classes: 4
        pretrained: false
        in_channels: r
      checkpoint: ../checkpoint-epoch66.pth

    b5-1019_190940:
      type: EffNet
      args:
        encoder_name: efficientnet-b4
        classes: 4
        pretrained: false
        in_channels: r
      checkpoint: ../checkpoint-epoch63.pth

segmentation:

  models:

    # LB 0.91489
    unet-se_resnext50_32x4d-1017_210727-LB-0.91489:
      weight: 0.95
      type: Unet
      args:
        encoder_name: se_resnext50_32x4d
        classes: 4
        encoder_weights: null
        in_channels: 1
        activation: null
      checkpoint: ../checkpoint-epoch128.pth

    # LB 0.91515
    unet-b5-1016_202517-LB-0.91515:
      weight: 1.0
      type: Unet
      args:
        encoder_name: efficientnet-b5
        classes: 4
        encoder_weights: imagenet
        in_channels: 1
        activation: null
      checkpoint: ../checkpoint-epoch151.pth

    # LB 0.91528
    fpn-b5-1016_221333-LB-0.91528:
      weight: 1.0
      type: FPN
      args:
        encoder_name: efficientnet-b5
        classes: 4
        encoder_weights: imagenet
        in_channels: 1
        activation: null
        weight_std: false
        decoder_merge_policy: cat
      checkpoint: ../checkpoint-epoch119.pth

    # LB 0.91393
    fpn-b5-1016_203538-LB-0.91393:
      weight: 0.85
      type: FPN
      args:
        encoder_name: efficientnet-b5
        classes: 4
        encoder_weights: imagenet
        in_channels: 1
        activation: null
        weight_std: false
        decoder_merge_policy: cat
      checkpoint: ../checkpoint-epoch133.pth

    # LB 0.91656
    fpn-se_resnext50_32x4d-1018_082516-LB-0.91656:
      weight: 1.1
      type: FPN
      args:
        encoder_name: se_resnext50_32x4d
        classes: 4
        encoder_weights: null
        in_channels: 1
        activation: null
        weight_std: false
        decoder_merge_policy: cat
      checkpoint: ../checkpoint-epoch186.pth

testing:
  data_dir: ../../understandingclouds_data/test_images
  batch_size: 8
  num_workers: 2
  verbose: 2

postprocess:
  args:
    domination: false
    dilation: 0
    p_thresh: [0.5, 0.5, 0.5, 0.5]
    min_class_sizes: [500, 600, 900, 1800]
    classification_factor: 10
"""

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------

config = yaml.safe_load(config_str)

df = Runner().predict(config)
print(config)
print(df.head(20))
df.to_csv("submission.csv", index=False)
print(f'Finished saving predictions to "submission.csv"')