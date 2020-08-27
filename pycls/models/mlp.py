#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MLP model."""

import torch.nn as nn
import torch

from pycls.config import cfg

import pycls.utils.logging as lu
import pycls.utils.net as nu
from .relation_graph import *

import time
import pdb

logger = lu.get_logger(__name__)


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        ##### (1) Level 1: channel
        'linear_transform': LinearTransform,
        'symlinear_transform': SymLinearTransform,
        'grouplinear_transform': GroupLinearTransform,
        'groupshufflelinear_transform': GroupShuffleLinearTransform,
        'talklinear_transform': TalkLinearTransform,  # relational graph
    }
    assert name in trans_funs.keys(), \
        'Transformation function \'{}\' not supported'.format(name)
    return trans_funs[name]


##### (0) Basic

class LinearTransform(nn.Module):
    """Basic transformation: linear"""

    def __init__(self, dim_in, dim_out, seed=None):
        super(LinearTransform, self).__init__()
        self._construct_class(dim_in, dim_out)

    def _construct_class(self, dim_in, dim_out):
        # 3x3, BN, ReLU
        self.a = nn.Linear(
            dim_in, dim_out, bias=False
        )
        self.a_bn = nn.BatchNorm1d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_bn.final_bn = True
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SymLinearTransform(nn.Module):
    """Basic transformation: linear, symmetric"""

    def __init__(self, dim_in, dim_out, seed=None):
        super(SymLinearTransform, self).__init__()
        self._construct_class(dim_in, dim_out)

    def _construct_class(self, dim_in, dim_out):
        # 3x3, BN, ReLU
        self.a = SymLinear(
            dim_in, dim_out, bias=False
        )
        self.a_bn = nn.BatchNorm1d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_bn.final_bn = True
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class GroupLinearTransform(nn.Module):
    """Basic transformation: linear, group"""

    def __init__(self, dim_in, dim_out, seed=None):
        super(GroupLinearTransform, self).__init__()
        self._construct_class(dim_in, dim_out)

    def _construct_class(self, dim_in, dim_out):
        # 3x3, BN, ReLU
        self.a = GroupLinear(
            dim_in, dim_out, bias=False, group_size=cfg.RGRAPH.GROUP_SIZE
        )
        self.a_bn = nn.BatchNorm1d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_bn.final_bn = True
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class GroupShuffleLinearTransform(nn.Module):
    """Basic transformation: linear, shuffle"""

    def __init__(self, dim_in, dim_out, seed=None):
        super(GroupShuffleLinearTransform, self).__init__()
        self._construct_class(dim_in, dim_out)

    def _construct_class(self, dim_in, dim_out):
        # 3x3, BN, ReLU
        self.a = GroupLinear(
            dim_in, dim_out, bias=False, group_size=cfg.RGRAPH.GROUP_SIZE
        )
        self.a_bn = nn.BatchNorm1d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_bn.final_bn = True
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.shuffle_shape = (dim_out // cfg.RGRAPH.GROUP_NUM, cfg.RGRAPH.GROUP_NUM)

    def forward(self, x):
        x = self.a(x)
        x = x.view(x.shape[0], self.shuffle_shape[0], self.shuffle_shape[1]).permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.a_bn(x)
        x = self.relu(x)
        return x


class TalkLinearTransform(nn.Module):
    """Basic transformation: linear, relational graph"""

    def __init__(self, dim_in, dim_out, seed=None):
        self.seed = seed
        super(TalkLinearTransform, self).__init__()
        self._construct_class(dim_in, dim_out)

    def _construct_class(self, dim_in, dim_out):
        self.a = TalkLinear(
            dim_in, dim_out, cfg.RGRAPH.GROUP_NUM, bias=False,
            message_type=cfg.RGRAPH.MESSAGE_TYPE, sparsity=cfg.RGRAPH.SPARSITY,
            p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed)

        self.a_bn = nn.BatchNorm1d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_bn.final_bn = True
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class MLPStage(nn.Module):
    """Stage of MLPNet."""

    def __init__(
            self, dim_in, dim_out, num_bs):
        super(MLPStage, self).__init__()
        self._construct_class(dim_in, dim_out, num_bs)

    def _construct_class(self, dim_in, dim_out, num_bs):
        if cfg.RGRAPH.KEEP_GRAPH:
            seed = cfg.RGRAPH.SEED_GRAPH
        else:
            seed = int(dim_out * 100 * cfg.RGRAPH.SPARSITY)
        for i in range(num_bs):
            b_dim_in = dim_in if i == 0 else dim_out
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            res_block = trans_fun(
                b_dim_in, dim_out, seed=seed
            )
            if not cfg.RGRAPH.KEEP_GRAPH:
                seed += 1
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class MLPStem(nn.Module):
    """Stem of MLPNet."""

    def __init__(self, dim_in, dim_out):
        super(MLPStem, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar(dim_in, dim_out)
        else:
            raise NotImplementedError

    def _construct_cifar(self, dim_in, dim_out):
        self.linear = nn.Linear(
            dim_in, dim_out, bias=False
        )
        self.bn = nn.BatchNorm1d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.children():
            x = layer(x)
        return x


class MLPHead(nn.Module):
    """MLPNet head."""

    def __init__(self, dim_in, num_classes):
        super(MLPHead, self).__init__()
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPNet(nn.Module):
    """MLPNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10'], \
            'Training MLPNet on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['cifar10'], \
            'Testing MLPNet on {} is not supported'.format(cfg.TEST.DATASET)
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(MLPNet, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar()
        else:
            raise NotImplementedError

        self.apply(nu.init_weights)

    # ##### basic transform
    def _construct_cifar(self):
        num_layers = cfg.MODEL.LAYERS
        dim_inner = cfg.RGRAPH.DIM_LIST[0]
        dim_first = cfg.RGRAPH.DIM_FIRST
        self.s1 = MLPStem(dim_in=3072, dim_out=dim_first)
        self.s2 = MLPStage(dim_in=dim_first, dim_out=dim_inner, num_bs=num_layers)
        self.head = MLPHead(dim_in=dim_inner, num_classes=cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
