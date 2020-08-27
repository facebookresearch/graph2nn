#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet models."""

import math

import torch
import torch.nn as nn

from pycls.config import cfg
import pycls.utils.net as nu

import pycls.utils.logging as logging

from .relation_graph import *

logger = logging.get_logger(__name__)


def get_conv(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        'mbconv_transform': MBConv,
        'mbtalkconv_transform': MBTalkConv,
    }
    assert name in trans_funs.keys(), \
        'Transformation function \'{}\' not supported'.format(name)
    return trans_funs[name]

def drop_connect_tf(x, drop_ratio):
    """Drop connect (tensorflow port)."""
    keep_ratio = 1.0 - drop_ratio
    rt = torch.rand([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    rt.add_(keep_ratio)
    bt = torch.floor(rt)
    x.div_(keep_ratio)
    x.mul_(bt)
    return x


def drop_connect_pt(x, drop_ratio):
    """Drop connect (pytorch version)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


def get_act_fun(act_type):
    """Retrieves the activations function."""
    act_funs = {
        'swish': Swish,
        'relu': nn.ReLU,
    }
    assert act_type in act_funs.keys(), \
        'Activation function \'{}\' not supported'.format(act_type)
    return act_funs[act_type]


class SimpleHead(nn.Module):
    """Simple head."""

    def __init__(self, dim_in, num_classes):
        super(SimpleHead, self).__init__()
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout
        if cfg.EFFICIENT_NET.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EFFICIENT_NET.DROPOUT_RATIO)
        # FC
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, 'dropout') else x
        x = self.fc(x)
        return x


class ConvHead(nn.Module):
    """EfficientNet conv head."""

    def __init__(self, in_w, out_w, num_classes, act_fun):
        super(ConvHead, self).__init__()
        self._construct_class(in_w, out_w, num_classes, act_fun)

    def _construct_class(self, in_w, out_w, num_classes, act_fun):
        # 1x1, BN, Swish
        self.conv = nn.Conv2d(
            in_w, out_w,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv_bn = nn.BatchNorm2d(
            out_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.conv_swish = act_fun()
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout
        if cfg.EFFICIENT_NET.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EFFICIENT_NET.DROPOUT_RATIO)
        # FC
        self.fc = nn.Linear(out_w, num_classes, bias=True)

    def forward(self, x):
        # 1x1, BN, Swish
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        # AvgPool
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # Dropout
        x = self.dropout(x) if hasattr(self, 'dropout') else x
        # FC
        x = self.fc(x)
        return x

class LinearHead(nn.Module):
    """EfficientNet linear head."""

    def __init__(self, in_w, out_w, num_classes, act_fun):
        super(LinearHead, self).__init__()
        self._construct_class(in_w, out_w, num_classes, act_fun)

    def _construct_class(self, in_w, out_w, num_classes, act_fun):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC0        
        self.fc0 = nn.Linear(in_w, out_w, bias=False)
        self.fc0_bn = nn.BatchNorm1d(
            out_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.fc0_swish = act_fun()        
        # FC
        self.fc = nn.Linear(out_w, num_classes, bias=True)

    def forward(self, x):
        # AvgPool
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # Linear, BN, Swish
        x = self.fc0_swish(self.fc0_bn(self.fc0(x)))
        # FC
        x = self.fc(x)
        return x


class MBConv(nn.Module):
    """Mobile inverted bottleneck block with SE (MBConv)."""

    def __init__(self, in_w, exp_r, kernel, stride, se_r, out_w, act_fun, seed=None, exp_w=None):
        super(MBConv, self).__init__()
        self._construct_class(in_w, exp_r, kernel, stride, se_r, out_w, act_fun)

    def _construct_class(self, in_w, exp_r, kernel, stride, se_r, out_w, act_fun):
        # Expansion: 1x1, BN, Swish
        self.expand = None
        exp_w = int(in_w * exp_r)
        # Include exp ops only if the exp ratio is different from 1
        if exp_w != in_w:
            self.expand = nn.Conv2d(
                in_w, exp_w,
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.expand_bn = nn.BatchNorm2d(
                exp_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
            )
            self.expand_swish = act_fun()
        # Depthwise: 3x3 dwise, BN, Swish
        self.dwise = nn.Conv2d(
            exp_w, exp_w,
            kernel_size=kernel, stride=stride, groups=exp_w, bias=False,
            # Hacky padding to preserve res  (supports only 3x3 and 5x5)
            padding=(1 if kernel == 3 else 2)
        )
        self.dwise_bn = nn.BatchNorm2d(
            exp_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.dwise_swish = act_fun()
        # SE: x * F_ex(x)
        if cfg.EFFICIENT_NET.SE_ENABLED:
            se_w = int(in_w * se_r)
            self.se = SE(exp_w, se_w, act_fun)
        # Linear projection: 1x1, BN
        self.lin_proj = nn.Conv2d(
            exp_w, out_w,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lin_proj_bn = nn.BatchNorm2d(
            out_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        # Nonlinear projection
        if not cfg.EFFICIENT_NET.LIN_PROJ:
            self.lin_proj_swish = act_fun()
        # Skip connections on blocks w/ same in and out shapes (MN-V2, Fig. 4)
        self.has_skip = (stride == 1) and (in_w == out_w)

    def forward(self, x):
        f_x = x
        # Expansion
        if self.expand:
            f_x = self.expand_swish(self.expand_bn(self.expand(f_x)))
        # Depthwise
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        # SE
        if cfg.EFFICIENT_NET.SE_ENABLED:
            f_x = self.se(f_x)
        # Linear projection
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        # Nonlinear projection
        if not cfg.EFFICIENT_NET.LIN_PROJ:
            f_x = self.lin_proj_swish(f_x)
        # Skip connection
        if self.has_skip:
            # Drop connect
            if self.training and cfg.EFFICIENT_NET.DC_RATIO > 0.0:
                if cfg.EFFICIENT_NET.DC_IMP == 'tf':
                    f_x = drop_connect_tf(f_x, cfg.EFFICIENT_NET.DC_RATIO)
                else:
                    f_x = drop_connect_pt(f_x, cfg.EFFICIENT_NET.DC_RATIO)
            f_x = x + f_x
        return f_x


class MBTalkConv(nn.Module):
    """Mobile inverted bottleneck block with SE (MBConv)."""

    def __init__(self, in_w, exp_r, kernel, stride, se_r, out_w, act_fun, seed=None, exp_w=None):
        super(MBTalkConv, self).__init__()
        self.seed=seed
        self._construct_class(in_w, exp_r, kernel, stride, se_r, out_w, act_fun, exp_w)

    def _construct_class(self, in_w, exp_r, kernel, stride, se_r, out_w, act_fun, exp_w):
        # Expansion: 1x1, BN, Swish
        self.expand = None

        if int(exp_r)==1:
            exp_w = in_w
        else:
            self.expand = TalkConv2d(
                in_w, exp_w, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
                stride=1, padding=0, bias=False,
                message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
                sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
            )
            self.expand_bn = nn.BatchNorm2d(
                exp_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
            )
            self.expand_swish = act_fun()
        # Depthwise: 3x3 dwise, BN, Swish
        self.dwise = nn.Conv2d(
            exp_w, exp_w,
            kernel_size=kernel, stride=stride, groups=exp_w, bias=False,
            # Hacky padding to preserve res  (supports only 3x3 and 5x5)
            padding=(1 if kernel == 3 else 2)
        )
        self.dwise_bn = nn.BatchNorm2d(
            exp_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.dwise_swish = act_fun()
        # SE: x * F_ex(x)
        if cfg.EFFICIENT_NET.SE_ENABLED:
            se_w = int(in_w * se_r)
            self.se = SE(exp_w, se_w, act_fun)
        # Linear projection: 1x1, BN
        self.lin_proj = TalkConv2d(
                exp_w, out_w, cfg.RGRAPH.GROUP_NUM, kernel_size=1,
                stride=1, padding=0, bias=False,
                message_type=cfg.RGRAPH.MESSAGE_TYPE, directed=cfg.RGRAPH.DIRECTED, agg=cfg.RGRAPH.AGG_FUNC,
                sparsity=cfg.RGRAPH.SPARSITY, p=cfg.RGRAPH.P, talk_mode=cfg.RGRAPH.TALK_MODE, seed=self.seed
            )
        self.lin_proj_bn = nn.BatchNorm2d(
            out_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        # Nonlinear projection
        if not cfg.EFFICIENT_NET.LIN_PROJ:
            self.lin_proj_swish = act_fun()
        # Skip connections on blocks w/ same in and out shapes (MN-V2, Fig. 4)
        self.has_skip = (stride == 1) and (in_w == out_w)

    def forward(self, x):
        f_x = x
        # Expansion
        if self.expand:
            f_x = self.expand_swish(self.expand_bn(self.expand(f_x)))
        # Depthwise
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        # SE
        if cfg.EFFICIENT_NET.SE_ENABLED:
            f_x = self.se(f_x)
        # Linear projection
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        # Nonlinear projection
        if not cfg.EFFICIENT_NET.LIN_PROJ:
            f_x = self.lin_proj_swish(f_x)
        # Skip connection
        if self.has_skip:
            # Drop connect
            if self.training and cfg.EFFICIENT_NET.DC_RATIO > 0.0:
                if cfg.EFFICIENT_NET.DC_IMP == 'tf':
                    f_x = drop_connect_tf(f_x, cfg.EFFICIENT_NET.DC_RATIO)
                else:
                    f_x = drop_connect_pt(f_x, cfg.EFFICIENT_NET.DC_RATIO)
            f_x = x + f_x
        return f_x







class Stage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, in_w, exp_r, kernel, stride, se_r, out_w, d, act_fun, exp_w=None):
        super(Stage, self).__init__()
        self._construct_class(in_w, exp_r, kernel, stride, se_r, out_w, d, act_fun, exp_w)

    def _construct_class(self, in_w, exp_r, kernel, stride, se_r, out_w, d, act_fun, exp_w):
        if cfg.RGRAPH.KEEP_GRAPH:
            seed = cfg.RGRAPH.SEED_GRAPH
        else:
            seed = int(cfg.RGRAPH.SEED_GRAPH*100)
        # Construct a sequence of blocks
        for i in range(d):
            trans_fun = get_conv(cfg.RESNET.TRANS_FUN)
            # Stride and input width apply to the first block of the stage
            stride_b = stride if i == 0 else 1
            in_w_b = in_w if i == 0 else out_w
            # Construct the block
            self.add_module(
                'b{}'.format(i + 1),
                trans_fun(in_w_b, exp_r, kernel, stride_b, se_r, out_w, act_fun, seed=seed, exp_w=exp_w)
            )
            if not cfg.RGRAPH.KEEP_GRAPH:
                seed += 1

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet."""

    def __init__(self, in_w, out_w, act_fun):
        super(StemIN, self).__init__()
        self._construct_class(in_w, out_w, act_fun)

    def _construct_class(self, in_w, out_w, act_fun):
        self.conv = nn.Conv2d(
            in_w, out_w,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_w, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.swish = act_fun()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class EfficientNet(nn.Module):
    """EfficientNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['imagenet'], \
            'Training on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['imagenet'], \
            'Testing on {} is not supported'.format(cfg.TEST.DATASET)
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        assert cfg.EFFICIENT_NET.HEAD_TYPE in ['conv_head', 'simple_head', 'linear_head'], \
            'Unsupported head type: {}'.format(cfg.EFFICIENT_NET.HEAD_TYPE)
        super(EfficientNet, self).__init__()
        self._construct_class(
            stem_w=cfg.EFFICIENT_NET.STEM_W,
            ds=cfg.EFFICIENT_NET.DEPTHS,
            ws=cfg.EFFICIENT_NET.WIDTHS,
            exp_rs=cfg.EFFICIENT_NET.EXP_RATIOS,
            se_r=cfg.EFFICIENT_NET.SE_RATIO,
            ss=cfg.EFFICIENT_NET.STRIDES,
            ks=cfg.EFFICIENT_NET.KERNELS,
            head_type=cfg.EFFICIENT_NET.HEAD_TYPE,
            head_w=cfg.EFFICIENT_NET.HEAD_W,
            act_type=cfg.EFFICIENT_NET.ACT_FUN,
            nc=cfg.MODEL.NUM_CLASSES
        )
        self.apply(nu.init_weights)

    def _construct_class(
        self, stem_w, ds, ws, exp_rs, se_r, ss, ks,
        head_type, head_w, act_type, nc
    ):
        """Constructs imagenet models."""
        # Group params by stage
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        # Activation function
        act_fun = get_act_fun(act_type)
        # Set dim for each stage
        dim_list = cfg.RGRAPH.DIM_LIST
        expdim_list = [int(cfg.EFFICIENT_NET.WIDTHS[i]*cfg.EFFICIENT_NET.EXP_RATIOS[i])
                       for i in range(len(cfg.EFFICIENT_NET.WIDTHS))]
        # Construct the stems
        self.stem = StemIN(3, stem_w, act_fun)
        prev_w = stem_w
        # Construct the stages
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            if cfg.RESNET.TRANS_FUN != 'mbconv_transform':
                w = dim_list[i]
            exp_w = expdim_list[i]
            self.add_module(
                's{}'.format(i + 1),
                Stage(prev_w, exp_r, kernel, stride, se_r, w, d, act_fun, exp_w=exp_w)
            )
            prev_w = w
        # Construct the head
        if head_type == 'conv_head':
            self.head = ConvHead(prev_w, head_w, nc, act_fun)
        elif head_type == 'linear_head':
            self.head = LinearHead(prev_w, head_w, nc, act_fun)
        else:
            self.head = SimpleHead(prev_w, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x