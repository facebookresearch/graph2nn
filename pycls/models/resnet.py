#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ResNe(X)t model."""

import torch.nn as nn
import torch

from pycls.config import cfg

import pycls.utils.logging as lu
import pycls.utils.net as nu
from .relation_graph import *

import time
import pdb

logger = lu.get_logger(__name__)

# Stage depths for an ImageNet model {model depth -> (d2, d3, d4, d5)}
_IN_MODEL_STAGE_DS = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        ############ Res-34
        'channelbasic_transform': ChannelBasicTransform,
        'groupbasictalk_transform': GroupBasicTalkTransform,
        ############ Res-34-sep
        'channelsep_transform': ChannelSepTransform,
        'groupseptalk_transform': GroupSepTalkTransform,
        ############ Res-50
        'bottleneck_transform': BottleneckTransform,
        'talkbottleneck_transform': TalkBottleneckTransform,

    }
    assert name in trans_funs.keys(), \
        'Transformation function \'{}\' not supported'.format(name)
    return trans_funs[name]


############ Res-34

class ChannelBasicTransform(nn.Module):
    """Basic transformation: 3x3, 3x3"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        super(ChannelBasicTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN
        self.b = nn.Conv2d(
            dim_out, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class GroupBasicTalkTransform(nn.Module):
    """Basic transformation: 3x3, 3x3, relational graph"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        self.seed = seed
        super(GroupBasicTalkTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = TalkConv2d(
            dim_in, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=3,
            stride=stride, padding=1, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.a_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN
        self.b = TalkConv2d(
            dim_out, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=3,
            stride=1, padding=1, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.b_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


############ Res-34-sep

class ChannelSepTransform(nn.Module):
    """Separable transformation: 3x3, 3x3"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        super(ChannelSepTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # ReLU, 3x3, BN, 1x1, BN
        self.a_3x3 = nn.Conv2d(
            dim_in, dim_in, kernel_size=3,
            stride=stride, padding=1, bias=False, groups=dim_in
        )
        self.a_1x1 = nn.Conv2d(
            dim_in, dim_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.a_1x1_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # ReLU, 3x3, BN, 1x1, BN
        self.b_3x3 = nn.Conv2d(
            dim_out, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False, groups=dim_out
        )
        self.b_1x1 = nn.Conv2d(
            dim_out, dim_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.b_1x1_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_1x1_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class GroupSepTalkTransform(nn.Module):
    """Separable transformation: 3x3, 3x3, relational graph"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        self.seed = seed
        super(GroupSepTalkTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # ReLU, 3x3, BN, 1x1, BN
        self.a_3x3 = nn.Conv2d(
            dim_in, dim_in, kernel_size=3,
            stride=stride, padding=1, bias=False, groups=dim_in
        )
        self.a_1x1 = TalkConv2d(
            dim_in, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
            stride=1, padding=0, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.a_1x1_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # ReLU, 3x3, BN, 1x1, BN
        self.b_3x3 = nn.Conv2d(
            dim_out, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False, groups=dim_out
        )
        self.b_1x1 = TalkConv2d(
            dim_out, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
            stride=1, padding=0, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.b_1x1_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_1x1_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


############ Res-50

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        super(BottleneckTransform, self).__init__()
        dim_inner = int(round(dim_out / 4))
        self._construct_class(dim_in, dim_out, stride, dim_inner, num_gs, seed)

    def _construct_class(self, dim_in, dim_out, stride, dim_inner, num_gs, seed):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        # (str1x1, str3x3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        (str1x1, str3x3) = (1, stride)

        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_inner, kernel_size=1,
            stride=str1x1, padding=0, bias=False
        )
        self.a_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            dim_inner, dim_inner, kernel_size=3,
            stride=str3x3, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 1x1, BN
        self.c = nn.Conv2d(
            dim_inner, dim_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.c_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class TalkBottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1, relational graph"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        super(TalkBottleneckTransform, self).__init__()
        dim_inner = int(round(dim_out / 4))
        self.seed = seed
        self._construct_class(dim_in, dim_out, stride, dim_inner, num_gs, seed)

    def _construct_class(self, dim_in, dim_out, stride, dim_inner, num_gs, seed):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        # (str1x1, str3x3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        (str1x1, str3x3) = (1, stride)

        # 1x1, BN, ReLU
        self.a = TalkConv2d(
            dim_in, dim_inner, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
            stride=str1x1, padding=0, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.a_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN, ReLU
        self.b = TalkConv2d(
            dim_inner, dim_inner, cfg.RGRAPH.GROUP_NUM, kernel_size=3,
            stride=str3x3, padding=1, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.b_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 1x1, BN
        self.c = TalkConv2d(
            dim_inner, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
            stride=1, padding=0, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
            sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
        )
        self.c_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


##### Remaining ResNet code

class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(
            self, dim_in, dim_out, stride, trans_fun, dim_inner=None, num_gs=1, seed=None):
        super(ResBlock, self).__init__()
        self.seed = seed
        self._construct_class(dim_in, dim_out, stride, trans_fun, dim_inner, num_gs, seed)

    def _add_skip_proj(self, dim_in, dim_out, stride):
        if 'group' in cfg.RESNET.TRANS_FUN and 'share' not in cfg.RESNET.TRANS_FUN:
            self.proj = TalkConv2d(
                dim_in, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
                stride=stride, padding=0, bias=False,
                message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
                sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
            )
        else:
            self.proj = nn.Conv2d(
                dim_in, dim_out, kernel_size=1,
                stride=stride, padding=0, bias=False
            )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

    def _construct_class(self, dim_in, dim_out, stride, trans_fun, dim_inner, num_gs, seed):
        # Use skip connection with projection if dim or res change
        self.proj_block = (dim_in != dim_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(dim_in, dim_out, stride)
        self.f = trans_fun(dim_in, dim_out, stride, dim_inner, num_gs, seed)
        self.act = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.act(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(
            self, dim_in, dim_out, stride, num_bs, dim_inner=None, num_gs=1):
        super(ResStage, self).__init__()
        self._construct_class(dim_in, dim_out, stride, num_bs, dim_inner, num_gs)

    def _construct_class(self, dim_in, dim_out, stride, num_bs, dim_inner, num_gs):
        if cfg.RGRAPH.KEEP_GRAPH:
            seed = cfg.RGRAPH.SEED_GRAPH
        else:
            seed = int(cfg.RGRAPH.SEED_GRAPH * 100)
        for i in range(num_bs):
            # Stride and dim_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_dim_in = dim_in if i == 0 else dim_out
            # Retrieve the transformation function
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            # Construct the block
            res_block = ResBlock(
                b_dim_in, dim_out, b_stride, trans_fun, dim_inner, num_gs, seed=seed
            )
            if not cfg.RGRAPH.KEEP_GRAPH:
                seed += 1
            self.add_module('b{}'.format(i + 1), res_block)
            for j in range(cfg.RGRAPH.ADD_1x1):
                trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN + '1x1')
                # Construct the block
                res_block = ResBlock(
                    dim_out, dim_out, 1, trans_fun, dim_inner, num_gs, seed=seed
                )
                if not cfg.RGRAPH.KEEP_GRAPH:
                    seed += 1
                self.add_module('b{}_{}1x1'.format(i + 1, j + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStem(nn.Module):
    """Stem of ResNet."""

    def __init__(self, dim_in, dim_out):
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(ResStem, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar(dim_in, dim_out)
        else:
            self._construct_imagenet(dim_in, dim_out)

    def _construct_cifar(self, dim_in, dim_out):
        # 3x3, BN, ReLU
        # self.conv = nn.Conv2d(
        #     dim_in, dim_out, kernel_size=3,
        #     stride=1, padding=1, bias=False
        # )
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=7,
            stride=1, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def _construct_imagenet(self, dim_in, dim_out):
        # 7x7, BN, ReLU, pool
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResHead(nn.Module):
    """ResNet head."""

    def __init__(self, dim_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet'], \
            'Training ResNet on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['cifar10', 'imagenet'], \
            'Testing ResNet on {} is not supported'.format(cfg.TEST.DATASET)
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(ResNet, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar()
        else:
            self._construct_imagenet()
        self.apply(nu.init_weights)

    # # ##### basic transform
    def _construct_cifar(self):
        assert (cfg.MODEL.DEPTH - 2) % 6 == 0, \
            'Model depth should be of the format 6n + 2 for cifar'
        logger.info('Constructing: ResNet-{}, cifar10'.format(cfg.MODEL.DEPTH))

        # Each stage has the same number of blocks for cifar
        num_blocks = int((cfg.MODEL.DEPTH - 2) / 6)
        # length = num of stages (excluding stem and head)
        dim_list = cfg.RGRAPH.DIM_LIST
        # Stage 1: (N, 3, 32, 32) -> (N, 16, 32, 32)*8
        # self.s1 = ResStem(dim_in=3, dim_out=16)
        self.s1 = ResStem(dim_in=3, dim_out=64)
        # Stage 2: (N, 16, 32, 32) -> (N, 16, 32, 32)
        # self.s2 = ResStage(dim_in=16, dim_out=dim_list[0], stride=1, num_bs=num_blocks)
        self.s2 = ResStage(dim_in=64, dim_out=dim_list[0], stride=1, num_bs=num_blocks)
        # Stage 3: (N, 16, 32, 32) -> (N, 32, 16, 16)
        self.s3 = ResStage(dim_in=dim_list[0], dim_out=dim_list[1], stride=2, num_bs=num_blocks)
        # Stage 4: (N, 32, 16, 16) -> (N, 64, 8, 8)
        self.s4 = ResStage(dim_in=dim_list[1], dim_out=dim_list[2], stride=2, num_bs=num_blocks)
        # Head: (N, 64, 8, 8) -> (N, num_classes)
        self.head = ResHead(dim_in=dim_list[2], num_classes=cfg.MODEL.NUM_CLASSES)

    # smaller imagenet
    def _construct_imagenet(self):
        logger.info('Constructing: ResNet-{}, Imagenet'.format(cfg.MODEL.DEPTH))
        # Retrieve the number of blocks per stage (excluding base)
        (d2, d3, d4, d5) = _IN_MODEL_STAGE_DS[cfg.MODEL.DEPTH]
        # Compute the initial inner block dim
        dim_list = cfg.RGRAPH.DIM_LIST
        # print(dim_list)
        # Stage 1: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.s1 = ResStem(dim_in=3, dim_out=64)
        # Stage 2: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s2 = ResStage(
            dim_in=64, dim_out=dim_list[0], stride=1, num_bs=d2
        )
        # Stage 3: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s3 = ResStage(
            dim_in=dim_list[0], dim_out=dim_list[1], stride=2, num_bs=d3
        )
        # Stage 4: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s4 = ResStage(
            dim_in=dim_list[1], dim_out=dim_list[2], stride=2, num_bs=d4
        )
        # Stage 5: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s5 = ResStage(
            dim_in=dim_list[2], dim_out=dim_list[3], stride=2, num_bs=d5
        )
        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = ResHead(dim_in=dim_list[3], num_classes=cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
