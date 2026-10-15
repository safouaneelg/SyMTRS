"""Distributed training helpers.

Kept deliberately small: the package supports single-process and `torchrun` /
`mp.spawn` launched DDP, and nothing fancier.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def world_size_from_env() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank_from_env() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def setup_ddp(local_rank: Optional[int] = None, world_size: Optional[int] = None) -> int:
    """Initialise the process group if one is not already up.

    Returns the rank. Uses the env-var rendezvous that torchrun sets, and falls back
    to a single-node rendezvous for `mp.spawn` launches.
    """
    local_rank = local_rank_from_env() if local_rank is None else local_rank
    world_size = world_size_from_env() if world_size is None else world_size

    if not dist.is_available():
        return 0
    if dist.is_initialized():
        return dist.get_rank()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29510")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, device: torch.device) -> float:
    """Average a python float across ranks so logged metrics are not per-rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())
