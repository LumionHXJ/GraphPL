import os
import random
import socket
import torch
import torch.distributed as dist
import functools
from collections import defaultdict

def synchronize_communication(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        dist.barrier()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        dist.barrier()
        return result
    return wrapper

def distribute_clients_to_cuda(clients_sample, world_size):
    num_gpus = world_size
    gpu2client = defaultdict(list)
    for ci, c in enumerate(clients_sample):
        gpu2client[ci % num_gpus].append(c)
    return gpu2client

def setup(exp, rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = exp.flags.port
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)