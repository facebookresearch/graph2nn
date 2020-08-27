#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CNN model."""

import torch.nn as nn
import torch

from pycls.config import cfg

import pycls.utils.logging as lu
import pycls.utils.net as nu
from .relation_graph import *

logger = lu.get_logger(__name__)


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        ##### (1) Level 1: channel
        ### (1.1) Basic Conv
        'convbasic_transform': ConvBasicTransform,
        'symconvbasic_transform': SymConvBasicTransform,
        'convtalk_transform': ConvTalkTransform, # relational graph

    }
    assert name in trans_funs.keys(), \
        'Transformation function \'{}\' not supported'.format(name)
    return trans_funs[name]


##### (1) Level 1: channel
### (1.1) Basic Conv
class ConvBasicTransform(nn.Module):
    """Basic transformation: 3x3"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        super(ConvBasicTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.a_bn.final_bn = True
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SymConvBasicTransform(nn.Module):
    """Basic transformation: 3x3 conv, symmetric"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        super(SymConvBasicTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = SymConv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.a_bn.final_bn = True
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ConvTalkTransform(nn.Module):
    """Basic transformation: 3x3 conv, relational graph"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1, seed=None):
        self.seed = seed
        super(ConvTalkTransform, self).__init__()
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
        # self.a_bn.final_bn = True
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


##### Remaining CNN code

class CNNStage(nn.Module):
    """Stage of CNN."""

    def __init__(
            self, dim_in, dim_out, stride, num_bs, dim_inner=None, num_gs=1):
        super(CNNStage, self).__init__()
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
            res_block = trans_fun(
                b_dim_in, dim_out, b_stride, dim_inner, num_gs, seed=seed
            )
            if not cfg.RGRAPH.KEEP_GRAPH:
                seed += 1
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class CNNStem(nn.Module):
    """Stem of CNN."""

    def __init__(self, dim_in, dim_out):
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(CNNStem, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar(dim_in, dim_out)
        else:
            self._construct_imagenet(dim_in, dim_out)

    def _construct_cifar(self, dim_in, dim_out):
        # 3x3, BN, ReLU
        if cfg.RGRAPH.STEM_MODE == 'default':
            self.conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=3,
                stride=1, padding=1, bias=False
            )
            self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS,
                                     momentum=cfg.BN.MOM)
            self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        elif cfg.RGRAPH.STEM_MODE == 'downsample':
            self.conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=3,
                stride=1, padding=1, bias=False
            )
            self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS,
                                     momentum=cfg.BN.MOM)
            self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _construct_imagenet(self, dim_in, dim_out):
        # 3x3, BN, ReLU, pool
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class CNNHead(nn.Module):
    """CNN head."""

    def __init__(self, dim_in, num_classes):
        super(CNNHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    """CNN model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet'], \
            'Training ResNet on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['cifar10', 'imagenet'], \
            'Testing ResNet on {} is not supported'.format(cfg.TEST.DATASET)
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(CNN, self).__init__()
        self._construct()
        self.apply(nu.init_weights)

    # # ##### basic transform
    def _construct(self):
        # Each stage has the same number of blocks for cifar
        dim_list = cfg.RGRAPH.DIM_LIST
        num_bs = cfg.MODEL.LAYERS // 3

        self.s1 = CNNStem(dim_in=3, dim_out=cfg.RGRAPH.DIM_FIRST)
        self.s2 = CNNStage(dim_in=cfg.RGRAPH.DIM_FIRST, dim_out=dim_list[0], stride=2, num_bs=num_bs)
        self.s3 = CNNStage(dim_in=dim_list[0], dim_out=dim_list[1], stride=2, num_bs=num_bs)
        self.s4 = CNNStage(dim_in=dim_list[1], dim_out=dim_list[2], stride=2, num_bs=num_bs)
        self.head = CNNHead(dim_in=dim_list[2], num_classes=cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
