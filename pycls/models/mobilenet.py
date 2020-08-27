#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MobileNet example"""

import torch.nn as nn
import torch.nn.functional as F
from pycls.config import cfg

import pycls.utils.net as nu

from .relation_graph import *


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()
        if cfg.RGRAPH.KEEP_GRAPH:
            self.seed = cfg.RGRAPH.SEED_GRAPH
        else:
            self.seed = int(cfg.RGRAPH.SEED_GRAPH * 100)

        def conv_bn(dim_in, dim_out, stride):
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)
            )

        def get_conv(name, dim_in, dim_out):
            if not cfg.RGRAPH.KEEP_GRAPH:
                self.seed += 1
            if name == 'channelbasic_transform':
                return nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

            elif name == 'groupbasictalk_transform':
                return TalkConv2d(
                    dim_in, dim_out, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
                    stride=1, padding=0, bias=False,
                    message_type=cfg.RGRAPH.MESSAGE_TYPE,
                    directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
                    sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P,
                    talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
                )

        def conv_dw(dim_in, dim_out, stride):
            conv1x1 = get_conv(cfg.RESNET.TRANS_FUN, dim_in, dim_out)

            return nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, stride, 1, groups=dim_in,
                          bias=False),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),

                conv1x1,
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),
            )

        self.dim_list = cfg.RGRAPH.DIM_LIST
        # print(self.dim_list)
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, self.dim_list[1], 1),
            conv_dw(self.dim_list[1], self.dim_list[2], 2),
            conv_dw(self.dim_list[2], self.dim_list[2], 1),
            conv_dw(self.dim_list[2], self.dim_list[3], 2),
            conv_dw(self.dim_list[3], self.dim_list[3], 1),
            conv_dw(self.dim_list[3], self.dim_list[4], 2),
            conv_dw(self.dim_list[4], self.dim_list[4], 1),
            conv_dw(self.dim_list[4], self.dim_list[4], 1),
            conv_dw(self.dim_list[4], self.dim_list[4], 1),
            conv_dw(self.dim_list[4], self.dim_list[4], 1),
            conv_dw(self.dim_list[4], self.dim_list[4], 1),
            conv_dw(self.dim_list[4], self.dim_list[5], 2),
            conv_dw(self.dim_list[5], self.dim_list[5], 1),
        )
        self.fc = nn.Linear(self.dim_list[5], num_classes)

        self.apply(nu.init_weights)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, self.dim_list[5])
        x = self.fc(x)
        return x
