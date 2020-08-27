#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import argparse
import numpy as np
import os
import sys
import torch
import multiprocessing as mp
import math
import pdb

from pycls.config import assert_cfg
from pycls.config import cfg
from pycls.config import dump_cfg
from pycls.datasets import loader
from pycls.models import model_builder
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter

import pycls.models.losses as losses
import pycls.models.optimizer as optim
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu

import time
from datetime import datetime
from tensorboardX import SummaryWriter

logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See pycls/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
            (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def log_model_info(model, writer_eval=None):
    """Logs model info"""
    logger.info('Model:\n{}'.format(model))
    params = mu.params_count(model)
    flops = mu.flops_count(model)
    logger.info('Params: {:,}'.format(params))
    logger.info('Flops: {:,}'.format(flops))
    logger.info('Number of node: {:,}'.format(cfg.RGRAPH.GROUP_NUM))
    # logger.info('{}, {}'.format(params,flops))
    if writer_eval is not None:
        writer_eval.add_scalar('Params', params, 1)
        writer_eval.add_scalar('Flops', flops, 1)
    return params, flops


def train_epoch(
        train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, writer_train=None, params=0, flops=0,
        is_master=False):
    """Performs one epoch of training."""

    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.scaled_all_reduce(
                [loss, top1_err, top5_err]
            )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(
            top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch, writer_train, params, flops, is_master=is_master)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(test_loader, model, test_meter, cur_epoch, writer_eval=None, params=0, flops=0, is_master=False):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    # test_meter.log_epoch_stats(cur_epoch,writer_eval,params,flops)
    test_meter.log_epoch_stats(cur_epoch, writer_eval, params, flops, model, is_master=is_master)
    stats = test_meter.get_epoch_stats(cur_epoch)
    test_meter.reset()
    if cfg.RGRAPH.SAVE_GRAPH:
        adj_dict = nu.model2adj(model)
        adj_dict = {**adj_dict, 'top1_err': stats['top1_err']}
        os.makedirs('{}/graphs/{}'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN), exist_ok=True)
        np.savez('{}/graphs/{}/{}.npz'.format(cfg.OUT_DIR, cfg.RGRAPH.SEED_TRAIN, cur_epoch), **adj_dict)


def train_model(writer_train=None, writer_eval=None, is_master=False):
    """Trains the model."""
    # Fit flops/params
    if cfg.TRAIN.AUTO_MATCH and cfg.RGRAPH.SEED_TRAIN == cfg.RGRAPH.SEED_TRAIN_START:
        mode = 'flops'  # flops or params
        if cfg.TRAIN.DATASET == 'cifar10':
            pre_repeat = 15
            if cfg.MODEL.TYPE == 'resnet':  # ResNet20
                stats_baseline = 40813184
            elif cfg.MODEL.TYPE == 'mlpnet':  # 5-layer MLP. cfg.MODEL.LAYERS exclude stem and head layers
                if cfg.MODEL.LAYERS == 3:
                    if cfg.RGRAPH.DIM_LIST[0] == 256:
                        stats_baseline = 985600
                    elif cfg.RGRAPH.DIM_LIST[0] == 512:
                        stats_baseline = 2364416
                    elif cfg.RGRAPH.DIM_LIST[0] == 1024:
                        stats_baseline = 6301696
            elif cfg.MODEL.TYPE == 'cnn':
                if cfg.MODEL.LAYERS == 3:
                    if cfg.RGRAPH.DIM_LIST[0] == 512:
                        stats_baseline = 806884352
                    elif cfg.RGRAPH.DIM_LIST[0] == 16:
                        stats_baseline = 1216672
                elif cfg.MODEL.LAYERS == 6:
                    if '64d' in cfg.OUT_DIR:
                        stats_baseline = 48957952
                    elif '16d' in cfg.OUT_DIR:
                        stats_baseline = 3392128
        elif cfg.TRAIN.DATASET == 'imagenet':
            pre_repeat = 9
            if cfg.MODEL.TYPE == 'resnet':
                if 'basic' in cfg.RESNET.TRANS_FUN:  # ResNet34
                    stats_baseline = 3663761408
                elif 'sep' in cfg.RESNET.TRANS_FUN:  # ResNet34-sep
                    stats_baseline = 553614592
                elif 'bottleneck' in cfg.RESNET.TRANS_FUN:  # ResNet50
                    stats_baseline = 4089184256
            elif cfg.MODEL.TYPE == 'efficientnet':  # EfficientNet
                stats_baseline = 385824092
            elif cfg.MODEL.TYPE == 'cnn':  # CNN
                if cfg.MODEL.LAYERS == 6:
                    if '64d' in cfg.OUT_DIR:
                        stats_baseline = 166438912
        cfg.defrost()
        stats = model_builder.build_model_stats(mode)
        if stats != stats_baseline:
            # 1st round: set first stage dim
            for i in range(pre_repeat):
                scale = round(math.sqrt(stats_baseline / stats), 2)
                first = cfg.RGRAPH.DIM_LIST[0]
                ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                first = int(round(first * scale))
                cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                stats = model_builder.build_model_stats(mode)
            flag_init = 1 if stats < stats_baseline else -1
            step = 1
            while True:
                first = cfg.RGRAPH.DIM_LIST[0]
                ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                first += flag_init * step
                cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                stats = model_builder.build_model_stats(mode)
                flag = 1 if stats < stats_baseline else -1
                if stats == stats_baseline:
                    break
                if flag != flag_init:
                    if cfg.RGRAPH.UPPER == False:  # make sure the stats is SMALLER than baseline
                        if flag < 0:
                            first = cfg.RGRAPH.DIM_LIST[0]
                            ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                            first -= flag_init * step
                            cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                        break
                    else:
                        if flag > 0:
                            first = cfg.RGRAPH.DIM_LIST[0]
                            ratio_list = [dim / first for dim in cfg.RGRAPH.DIM_LIST]
                            first -= flag_init * step
                            cfg.RGRAPH.DIM_LIST = [int(round(first * ratio)) for ratio in ratio_list]
                        break
            # 2nd round: set other stage dim
            first = cfg.RGRAPH.DIM_LIST[0]
            ratio_list = [int(round(dim / first)) for dim in cfg.RGRAPH.DIM_LIST]
            stats = model_builder.build_model_stats(mode)
            flag_init = 1 if stats < stats_baseline else -1
            if 'share' not in cfg.RESNET.TRANS_FUN:
                for i in range(1, len(cfg.RGRAPH.DIM_LIST)):
                    for j in range(ratio_list[i]):
                        cfg.RGRAPH.DIM_LIST[i] += flag_init
                        stats = model_builder.build_model_stats(mode)
                        flag = 1 if stats < stats_baseline else -1
                        if flag_init != flag:
                            cfg.RGRAPH.DIM_LIST[i] -= flag_init
                            break
        stats = model_builder.build_model_stats(mode)
        print('FINAL', cfg.RGRAPH.GROUP_NUM, cfg.RGRAPH.DIM_LIST, stats, stats_baseline, stats < stats_baseline)
    # Build the model (before the loaders to ease debugging)
    model = model_builder.build_model()
    params, flops = log_model_info(model, writer_eval)

    # Define the loss function
    loss_fun = losses.get_loss_fun()
    # Construct the optimizer
    optimizer = optim.construct_optimizer(model)

    # Load a checkpoint if applicable
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint():
        last_checkpoint = cu.get_checkpoint_last()
        checkpoint_epoch = cu.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info('Loaded checkpoint from: {}'.format(last_checkpoint))
        if checkpoint_epoch == cfg.OPTIM.MAX_EPOCH:
            exit()
            start_epoch = checkpoint_epoch
        else:
            start_epoch = checkpoint_epoch + 1

    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    test_meter = TestMeter(len(test_loader))

    if cfg.ONLINE_FLOPS:
        model_dummy = model_builder.build_model()

        IMAGE_SIZE = 224
        n_flops, n_params = mu.measure_model(model_dummy, IMAGE_SIZE, IMAGE_SIZE)

        logger.info('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))

        del (model_dummy)

    # Perform the training loop
    logger.info('Start epoch: {}'.format(start_epoch + 1))

    # do eval at initialization
    eval_epoch(test_loader, model, test_meter, -1,
               writer_eval, params, flops, is_master=is_master)

    if start_epoch == cfg.OPTIM.MAX_EPOCH:
        cur_epoch = start_epoch - 1
        eval_epoch(test_loader, model, test_meter, cur_epoch,
                   writer_eval, params, flops, is_master=is_master)
    else:
        for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
            # Train for one epoch
            train_epoch(
                train_loader, model, loss_fun, optimizer, train_meter, cur_epoch,
                writer_train, is_master=is_master
            )
            # Compute precise BN stats
            if cfg.BN.USE_PRECISE_STATS:
                nu.compute_precise_bn_stats(model, train_loader)
            # Save a checkpoint
            if cu.is_checkpoint_epoch(cur_epoch):
                checkpoint_file = cu.save_checkpoint(model, optimizer, cur_epoch)
                logger.info('Wrote checkpoint to: {}'.format(checkpoint_file))
            # Evaluate the model
            if is_eval_epoch(cur_epoch):
                eval_epoch(test_loader, model, test_meter, cur_epoch,
                           writer_eval, params, flops, is_master=is_master)


def single_proc_train():
    """Performs single process training."""

    # Setup logging
    lu.setup_logging()

    # Show the config
    logger.info('Config:\n{}'.format(cfg))
    # Setup tensorboard if provided
    writer_train = None
    writer_eval = None
    ## If use tensorboard
    if cfg.TENSORBOARD and du.is_master_proc() and cfg.RGRAPH.SEED_TRAIN == cfg.RGRAPH.SEED_TRAIN_START:
        comment = ''
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir_train = os.path.join(cfg.OUT_DIR,
                                    'runs', current_time + comment + '_train')
        logdir_eval = os.path.join(cfg.OUT_DIR,
                                   'runs', current_time + comment + '_eval')
        if not os.path.exists(logdir_train):
            os.makedirs(logdir_train)
        if not os.path.exists(logdir_eval):
            os.makedirs(logdir_eval)
        writer_train = SummaryWriter(logdir_train)
        writer_eval = SummaryWriter(logdir_eval)

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RGRAPH.SEED_TRAIN)
    torch.manual_seed(cfg.RGRAPH.SEED_TRAIN)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # Train the model
    train_model(writer_train, writer_eval, is_master=du.is_master_proc())

    if writer_train is not None and writer_eval is not None:
        writer_train.close()
        writer_eval.close()


def check_seed_exists(i):
    fname = "{}/results_epoch{}.txt".format(cfg.OUT_DIR, cfg.OPTIM.MAX_EPOCH)
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        if len(lines) > i:
            return True
    return False


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    # cfg.freeze()

    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the config
    dump_cfg()

    for i, cfg.RGRAPH.SEED_TRAIN in enumerate(range(cfg.RGRAPH.SEED_TRAIN_START, cfg.RGRAPH.SEED_TRAIN_END)):
        # check if a seed has been run
        if not check_seed_exists(i):
            if cfg.NUM_GPUS > 1:
                mpu.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=single_proc_train)
            else:
                single_proc_train()
        else:
            print('Seed {} exists, skip!'.format(cfg.RGRAPH.SEED_TRAIN))


if __name__ == '__main__':
    main()
