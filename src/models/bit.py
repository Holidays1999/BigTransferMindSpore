# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

import math
from collections import OrderedDict  # pylint: disable=g-importing-member

import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, context, ops, Tensor

from src.models.initializer import trunc_normal_, zeros_, ones_, kaiming_uniform_
from src.models.initializer import uniform_, _calculate_fan_in_and_fan_out


class StdConv2d(nn.Conv2d):

    def construct(self, x):
        weight = self.weight
        mean = ops.ReduceMean(True)(self.weight, (1, 2, 3))
        var = ops.Cast()(self.weight.var((1, 2, 3), 0, True), mean.dtype)
        weight_std = (weight - mean) / ops.Sqrt()(var + 1e-10)
        x = self.conv2d(x, weight_std)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        return x


def conv3x3(cin, cout, stride=1, group=1, has_bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     pad_mode='pad', has_bias=has_bias, group=group, padding=1)


def conv1x1(cin, cout, stride=1, has_bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     pad_mode='same', has_bias=has_bias)


class PreActBottleneck(nn.Cell):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU()
        self.downsample = None

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def construct(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual


class GlobalAveragePool(nn.Cell):
    def construct(self, x):
        return ops.ReduceMean(True)(x, [2, 3])


class ResNetV2(nn.Cell):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, num_classes=10, zero_head=True):
        super().__init__()
        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.SequentialCell(OrderedDict([
            ('conv',
             StdConv2d(3, 64 * width_factor, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=False)),
            ('pad', nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')),
        ]))

        self.body = nn.SequentialCell(OrderedDict([
            ('block1', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=64 * width_factor, cout=256 * width_factor, cmid=64 * width_factor))] +
                [(f'unit{i:02d}',
                  PreActBottleneck(cin=256 * width_factor, cout=256 * width_factor, cmid=64 * width_factor)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=256 * width_factor, cout=512 * width_factor, cmid=128 * width_factor,
                                             stride=2))] +
                [(f'unit{i:02d}',
                  PreActBottleneck(cin=512 * width_factor, cout=512 * width_factor, cmid=128 * width_factor)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=512 * width_factor, cout=1024 * width_factor, cmid=256 * width_factor,
                                             stride=2))] +
                [(f'unit{i:02d}',
                  PreActBottleneck(cin=1024 * width_factor, cout=1024 * width_factor, cmid=256 * width_factor)) for i in
                 range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.SequentialCell(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024 * width_factor, cout=2048 * width_factor, cmid=512 * width_factor,
                                             stride=2))] +
                [(f'unit{i:02d}',
                  PreActBottleneck(cin=2048 * width_factor, cout=2048 * width_factor, cmid=512 * width_factor)) for i in
                 range(2, block_units[3] + 1)],
            ))),
        ]))
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.SequentialCell(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048 * width_factor)),
            ('relu', nn.ReLU()),
            ('avg', GlobalAveragePool()),
            ('conv', nn.Conv2d(2048 * width_factor, num_classes, kernel_size=1, has_bias=True)),
        ]))
        self.init_weights()

    def construct(self, x):
        x = self.root(x)
        x = self.body(x)
        x = self.head(x)
        x = x.reshape(x.shape[0], -1)
        return x

    def init_weights(self):
        for name, cell in self.cells_and_names():
            init_weights(cell, name)
        if self.zero_head:
            zeros_(self.head.conv.weight)
            zeros_(self.head.conv.bias)


def init_weights(cell: nn.Cell, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (cell name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(cell, nn.Dense):
        trunc_normal_(cell.weigit, 0.02)
    elif isinstance(cell, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        kaiming_uniform_(cell.weight)
        if cell.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(cell.bias, bound)
    elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        ones_(cell.gamma)
        zeros_(cell.beta)


def BiT_M_R50x1(num_classes=1000, zero_head=True):
    return ResNetV2([3, 4, 6, 3], 1, num_classes=num_classes, zero_head=zero_head)


def BiT_M_R152x4(num_classes=1000, zero_head=True):
    return ResNetV2([3, 8, 36, 3], 4, num_classes=num_classes, zero_head=zero_head)


def BiT_M_R152x2(num_classes=1000, zero_head=True):
    return ResNetV2([3, 8, 36, 3], 2, num_classes=num_classes, zero_head=zero_head)


def BiT_M_R152x1(num_classes=1000, zero_head=True):
    return ResNetV2([3, 8, 36, 3], 2, num_classes=num_classes, zero_head=zero_head)


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])

if __name__ == "__main__":
    context.set_context(mode=0)
    model = BiT_M_R152x1()
    data = Tensor(np.random.randn(1, 3, 224, 224), dtype=mstype.float32)
    # out = model(data)
    # print(out.shape)
    params = 0.
    names = []
    for name, param in model.parameters_and_names():
        params += np.prod(param.shape)
        print(name, param.shape)
        names.append(name)
    with open("ms_name_152.txt", 'w') as f:
        for name in names:
            f.writelines(name + '\n')
    print(params, 25549352)
