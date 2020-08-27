#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for computing metrics."""

import numpy as np
import torch
import torch.nn as nn
import pdb
from pycls.config import cfg

from functools import reduce
import operator

from ..models.relation_graph import *
# Number of bytes in a megabyte
_B_IN_MB = 1024 * 1024


def topks_correct(preds, labels, ks):
    """Computes the number of top-k correct predictions for each k."""
    assert preds.size(0) == labels.size(0), \
        'Batch dim of predictions and labels must match'
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def params_count(model):
    """Computes the number of parameters."""
    count = 0
    for n,m in model.named_modules():
        if isinstance(m, TalkConv2d) or isinstance(m, TalkLinear):
            count += np.sum([p.numel()*m.params_scale for p in m.parameters(recurse=False)]).item()
        else:
            count += np.sum([p.numel() for p in m.parameters(recurse=False)]).item()
    return int(count)

def flops_count(model):
    """Computes the number of flops."""
    assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet'], \
        'Computing flops for {} is not supported'.format(cfg.TRAIN.DATASET)
    im_size = 32 if cfg.TRAIN.DATASET == 'cifar10' else 224
    h, w = im_size, im_size
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if '.se' in n:
                count += m.in_channels * m.out_channels + m.bias.numel()
                continue
            h_out = (h + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            w_out = (w + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
            count += np.prod([
                m.weight.numel(),
                h_out, w_out
            ])
            if 'proj' not in n:
                h, w = h_out, w_out
        elif isinstance(m, TalkConv2d):
            h_out = (h + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            w_out = (w + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
            count += int(np.prod([
                m.weight.numel()*m.flops_scale,
                h_out, w_out
            ]))
            if 'proj' not in n and 'pool' not in n:
                h, w = h_out, w_out
        elif isinstance(m, nn.MaxPool2d):
            h = (h + 2 * m.padding - m.kernel_size) // m.stride + 1
            w = (w + 2 * m.padding - m.kernel_size) // m.stride + 1
        elif isinstance(m, TalkLinear):
            count += int(m.in_features * m.out_features * m.flops_scale)
        elif isinstance(m, nn.Linear):
            count += m.in_features * m.out_features

    return count


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_MB

# Online FLOPs/Params calculation from CondenseNet codebase

count_ops = 0
count_params = 0

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        print(layer)
        print('out_h: ', out_h,  'out_w:', out_w)
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d', 'MaxPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    elif type_name in ['WeightedSumTransform']:
        weight_ops = layer.weight.numel() * multi_add
        delta_ops = x.size()[0] * (weight_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'Sigmoid', 'DirichletWeightedSumTransform', 'Softmax', 'Identity', 'Sequential']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = torch.zeros(1, 3, H, W).cuda()
    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params
