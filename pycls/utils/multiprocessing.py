#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multiprocessing helpers."""

import multiprocessing as mp
import traceback
import subprocess
import numpy as np

from pycls.utils.error_handler import ErrorHandler

import pycls.utils.distributed as du


def run(proc_rank, world_size, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        du.init_process_group(proc_rank, world_size)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except KeyboardInterrupt:
        # Killed by the parent process
        pass
    except Exception:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        # Destroy the process group
        du.destroy_process_group()


def multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs={}):
    """Runs a function in a multi-proc setting."""

    # Handle errors from training subprocesses
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Run each training subprocess
    ps = []
    for i in range(num_proc):
        p_i = mp.Process(
            target=run,
            args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)

    # Wait for each subprocess to finish
    for p in ps:
        p.join()


# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def auto_select_gpu(memory_threshold=7000, smooth_ratio=200):
    gpu_memory_raw = get_gpu_memory_map() + 10
    gpu_memory = gpu_memory_raw / smooth_ratio
    gpu_memory = gpu_memory.sum() / (gpu_memory + 10)
    gpu_memory[gpu_memory_raw > memory_threshold] = 0
    gpu_prob = gpu_memory / gpu_memory.sum()
    cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
    print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    return cuda
