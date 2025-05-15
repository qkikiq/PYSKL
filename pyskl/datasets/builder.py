# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import platform
import random
import torch
from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from .samplers import ClassSpecificDistributedSampler, DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    # 使用构建器函数 build_from_cfg 根据 cfg 中的配置，在注册表 DATASETS 中查找对应的数据集类型
    # 然后使用 cfg 和 default_args 创建该数据集的实例对象
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    在分布式训练中，每个 GPU（进程）都会有一个独立的 dataloader。
    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        videos_per_gpu (int): Number of videos on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.8.0.
            Default: False
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    # 如果数据集具有类别采样概率 class_prob，则使用类别特定的分布式采样器
    if hasattr(dataset, 'class_prob') and dataset.class_prob is not None:
        sampler = ClassSpecificDistributedSampler(
            dataset,
            world_size,
            rank,
            class_prob=dataset.class_prob,
            shuffle=shuffle,
            seed=seed)
    else:  # 否则使用常规的分布式采样器
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, seed=seed)
        # 分布式训练下，DataLoader 不再自行打乱顺序，由 sampler 控制
    shuffle = False
    # 每个 GPU 的 batch size
    batch_size = videos_per_gpu
    num_workers = workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,  #数据集
        batch_size=batch_size, #每个batch的样本数
        sampler=sampler,  # 使用分布式采样器
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=videos_per_gpu),  #用于将一个样本列表转换成一个批次数据
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
