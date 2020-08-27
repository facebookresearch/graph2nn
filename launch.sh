#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

TASK=${1:-mlp_cifar10}
DIVISION=${2:-best}
GPU=${3:-1}
OUT=${4:-stdout} # to analyze result, you should change from "stdout" to "file"

if [ "$TASK" = "mlp_cifar10" ]
then
    DATASET='cifar10'
    PREFIX='mlp_bs128_1gpu_layer3'
elif [ "$TASK" = "mlp_cifar10_bio" ]
then
    DATASET='cifar10'
    PREFIX='mlp_bs128_1gpu_layer3'
elif [ "$TASK" = "cnn_imagenet" ]
then
    DATASET='imagenet'
    if [ "GPU" = 1 ]
    then
    PREFIX='cnn6_bs32_1gpu_64d'
    else
    PREFIX='cnn6_bs256_8gpu_64d'
    fi
elif [ "$TASK" = "resnet34_imagenet" ]
then
    DATASET='imagenet'
    if [ "GPU" = 1 ]
    then
    PREFIX='R-34_bs32_1gpu'
    else
    PREFIX='R-34_bs256_8gpu'
    fi
elif [ "$TASK" = "resnet34sep_imagenet" ]
then
    DATASET='imagenet'
    if [ "GPU" = 1 ]
    then
    PREFIX='R-34_bs32_1gpu'
    else
    PREFIX='R-34_bs256_8gpu'
    fi
elif [ "$TASK" = "resnet50_imagenet" ]
then
    DATASET='imagenet'
    if [ "GPU" = 1 ]
    then
    PREFIX='R-50_bs32_1gpu'
    else
    PREFIX='R-50_bs256_8gpu'
    fi
elif [ "$TASK" = "efficient_imagenet" ]
then
    DATASET='imagenet'
    if [ "GPU" = 1 ]
    then
    PREFIX='EN-B0_bs64_1gpu_nms'
    else
    PREFIX='EN-B0_bs512_8gpu_nms'
    fi
else
   exit 1
fi

DIR=configs/baselines/${DATASET}/${TASK}/${DIVISION}/*

(trap 'kill 0' SIGINT;
for CONFIG in $DIR
do
    if echo "$CONFIG" | grep -q "$PREFIX"; then
        if [ "${CONFIG##*.}" = "yaml" ]; then
            CONFIG=${CONFIG##*/}
            CONFIG=${CONFIG%.yaml}
            echo ${CONFIG}
            # run one model at a time
            # Note: with slurm scheduler, one can run multiple jobs in parallel
            python tools/train_net.py --cfg configs/baselines/${DATASET}/${TASK}/${DIVISION}/${CONFIG}.yaml TRAIN.AUTO_RESUME False OUT_DIR checkpoint/${DATASET}/${TASK}/${DIVISION}/${CONFIG} BN.USE_PRECISE_STATS True LOG_DEST $OUT
        fi
    fi
done
)
