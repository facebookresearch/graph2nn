#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file."""


import os

from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type to use
_C.MODEL.TYPE = ''

# Number of weight layers
_C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = 'cross_entropy'

# Num layers, excluding the stem and head layers. Total layers used should +2
_C.MODEL.LAYERS = 3

# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
_C.RESNET = CN()

# Transformation function (see pycls/models/resnet.py for options)
_C.RESNET.TRANS_FUN = 'basic_transform'

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = False

# Whether append 1x1 resblock
_C.RESNET.APPEND1x1 = 0

# For group conv only
_C.RESNET.GROUP_SIZE = 2




# ---------------------------------------------------------------------------- #
# EfficientNet options
# ---------------------------------------------------------------------------- #
_C.EFFICIENT_NET = CN()

# Stem width
_C.EFFICIENT_NET.STEM_W = 32

# Depth for each stage (number of blocks in the stage)
_C.EFFICIENT_NET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.EFFICIENT_NET.WIDTHS = []

# Expansion ratios for MBConv blocks in each stage
_C.EFFICIENT_NET.EXP_RATIOS = []

# Squeeze-and-Excitation (SE) operation
_C.EFFICIENT_NET.SE_ENABLED = True

# Squeeze-and-Excitation (SE) ratio
_C.EFFICIENT_NET.SE_RATIO = 0.25

# Linear projection
_C.EFFICIENT_NET.LIN_PROJ = True

# Strides for each stage (applies to the first block of each stage)
_C.EFFICIENT_NET.STRIDES = []

# Kernel sizes for each stage
_C.EFFICIENT_NET.KERNELS = []

# Head type ('conv_head' or 'simple_head')
_C.EFFICIENT_NET.HEAD_TYPE = 'conv_head'

# Head width (applies to 'conv_head')
_C.EFFICIENT_NET.HEAD_W = 1280

# Ativation function
_C.EFFICIENT_NET.ACT_FUN = 'swish'

# Drop connect ratio
_C.EFFICIENT_NET.DC_RATIO = 0.0

# Drop connect implementation
_C.EFFICIENT_NET.DC_IMP = 'tf'

# Dropout ratio
_C.EFFICIENT_NET.DROPOUT_RATIO = 0.0



# ---------------------------------------------------------------------------- #
# Relational graph options
# ---------------------------------------------------------------------------- #
_C.RGRAPH = CN()


# dim for first layer. NOTE: this is fixed when matching FLOPs
_C.RGRAPH.DIM_FIRST = 16

# dim for each stage
_C.RGRAPH.DIM_LIST = []

# wide stem module
_C.RGRAPH.STEM_MODE = 'default'

# How to message exchange: dense, hier (deprecated)
_C.RGRAPH.TALK_MODE = 'dense'

# Num of nodes
_C.RGRAPH.GROUP_NUM = 32

# Size of nodes in Stage 1
_C.RGRAPH.GROUP_SIZE = 1

# The type of message passing used
_C.RGRAPH.MESSAGE_TYPE = 'ws'

# Whether use directed graph
_C.RGRAPH.DIRECTED = False

# Graph sparsity
_C.RGRAPH.SPARSITY = 0.5

# Graph Randomness
_C.RGRAPH.P = 0.0

# Graph seed
_C.RGRAPH.SEED_GRAPH = 1

# training seed used
_C.RGRAPH.SEED_TRAIN = 1

# training seed, start, end
_C.RGRAPH.SEED_TRAIN_START = 1
_C.RGRAPH.SEED_TRAIN_END = 2

# Keep graph across the network
_C.RGRAPH.KEEP_GRAPH = True

# Append additaion 1x1 layers for additional talks
_C.RGRAPH.ADD_1x1 = 0

# Match upper computational bound
_C.RGRAPH.UPPER = True

# Auto match computational budget
_C.RGRAPH.AUTO_MATCH = True

# AGG func. Only sum is supported in current mask-based implementation
_C.RGRAPH.AGG_FUNC = 'sum'

# Save weight matrices as graphs. Warning: the saved matrices can be huge
_C.RGRAPH.SAVE_GRAPH = False






# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = True
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = 'cos'

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs)
_C.OPTIM.STEP_SIZE = 1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Dataset and split
_C.TRAIN.DATASET = ''
_C.TRAIN.SPLIT = 'train'

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Checkpoint to start training from (if no automatic checkpoint saved)
_C.TRAIN.START_CHECKPOINT = ''

_C.TRAIN.AUTO_MATCH = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Dataset and split
_C.TEST.DATASET = ''
_C.TEST.SPLIT = 'val'

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = '/tmp'

# Config destination (in OUT_DIR)
_C.CFG_DEST = 'config.yaml'

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = 'stdout'

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = 'nccl'

# Hostname and port for initializing multi-process groups
_C.HOST = 'localhost'
_C.PORT = 12002

# Computing flops by online foward pass
_C.ONLINE_FLOPS = False


# Whether use Tensorboard
_C.TENSORBOARD = False


def assert_cfg():
    """Checks config values invariants."""
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, \
        'The first lr step must start at 0'
    assert _C.TRAIN.SPLIT in ['train', 'val', 'test'], \
        'Train split \'{}\' not supported'.format(_C.TRAIN.SPLIT)
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Train mini-batch size should be a multiple of NUM_GPUS.'
    assert _C.TEST.SPLIT in ['train', 'val', 'test'], \
        'Test split \'{}\' not supported'.format(_C.TEST.SPLIT)
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Test mini-batch size should be a multiple of NUM_GPUS.'
    # assert not _C.BN.USE_PRECISE_STATS or _C.NUM_GPUS == 1, \
    #     'Precise BN stats computation not verified for > 1 GPU'
    assert _C.LOG_DEST in ['stdout', 'file'], \
        'Log destination \'{}\' not supported'.format(_C.LOG_DEST)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, 'w') as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest='config.yaml'):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)
