#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""VGG example"""

import torch.nn as nn
import torch.nn.functional as F
from pycls.config import cfg

import pycls.utils.net as nu

from .relation_graph import *


class VGG(nn.Module):
    def __init__(self, num_classes=1024):
        super(VGG, self).__init__()
        self.seed = cfg.RGRAPH.SEED_GRAPH

        def conv_bn(dim_in, dim_out, stride, stem=False):
            if stem:
                conv = get_conv('convbasic_transform', dim_in, dim_out, stride)
            else:
                conv = get_conv(cfg.RESNET.TRANS_FUN, dim_in, dim_out, stride)
            return nn.Sequential(
                conv,
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)
            )

        def get_conv(name, dim_in, dim_out, stride=1):
            if not cfg.RGRAPH.KEEP_GRAPH:
                self.seed += 1
            if name == 'convbasic_transform':
                return nn.Conv2d(dim_in, dim_out,
                                 kernel_size=3, stride=stride,
                                 padding=1, bias=False)

            elif name == 'convtalk_transform':
                return TalkConv2d(
                    dim_in, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=3,
                    stride=stride, padding=1, bias=False,
                    message_type=cfg.RGRAPH.MESSAGE_TYPE,
                    directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
                    sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P,
                    talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
                )

        self.dim_list = cfg.RGRAPH.DIM_LIST
        # print(self.dim_list)
        self.model = nn.Sequential(
            conv_bn(3, 64, 1, stem=True),
            conv_bn(64, self.dim_list[0], 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn(self.dim_list[0], self.dim_list[1], 1),
            conv_bn(self.dim_list[1], self.dim_list[1], 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn(self.dim_list[1], self.dim_list[2], 1),
            conv_bn(self.dim_list[2], self.dim_list[2], 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn(self.dim_list[2], self.dim_list[3], 1),
            conv_bn(self.dim_list[3], self.dim_list[3], 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn(self.dim_list[3], self.dim_list[3], 1),
            conv_bn(self.dim_list[3], self.dim_list[3], 1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.dim_list[3], num_classes)

        self.apply(nu.init_weights)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
