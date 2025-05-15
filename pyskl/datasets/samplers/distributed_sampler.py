# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from collections import defaultdict
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from 'torch.utils.data.DistributedSampler'.
    样本按照指定的类别概率(class_prob)进行采样。这个采样器仅适用于单类别识别数据集。
    Samples are sampled with a class specific probability (class_prob). This sampler is only applicable to single class
       此采样器也与RepeatDataset兼容。
    recognition dataset. This sampler is also compatible with RepeatDataset.
    """

    def __init__(self,
                 dataset,  # 数据集对象
                 num_replicas=None,     #分布式训练的副本(进程)
                 rank=None,  # 当前进程的rank
                 class_prob=None,       # 类别采样概率
                 shuffle=True,  # 是否打乱数据
                 seed=0):  # 随机种子

        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        # 处理class_prob参数，确保其为字典类型
        if class_prob is not None:
            # 如果是列表，转换为字典{类别索引:概率}
            if isinstance(class_prob, list):
                class_prob = {i: n for i, n in enumerate(class_prob)}
               # 确保class_prob是字典类型
            assert isinstance(class_prob, dict)
        self.class_prob = class_prob
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self.seed + self.epoch)   #每轮使用不同的种子（与 epoch 相关）

        class_prob = self.class_prob
        dataset_name = type(self.dataset).__name__   # 获取数据集的名称
        dataset = self.dataset if dataset_name != 'RepeatDataset' else self.dataset.dataset  # 获取数据集对象
        times = 1
        if dataset_name == 'RepeatDataset':   # 如果数据集是 RepeatDataset，则获取原始数据集的 times 属性
            times = self.dataset.times # 将类别概率乘以重复次数，确保采样数量一致
            # 将类别概率乘以重复次数，保持采样一致性
            class_prob = {k: v * times for k, v in class_prob.items()}

        # 提取每个样本的标签，video_infos 是视频信息列表，包含 label 等字段
        labels = [x['label'] for x in dataset.video_infos]
        # 将索引按标签分组：samples[label] = [indices...]
        samples = defaultdict(list)
        for i, lb in enumerate(labels):
            samples[lb].append(i)

        indices = []  # 最终采样的索引集合
        for class_idx, class_indices in samples.items():
            mul = class_prob.get(class_idx, times)  # 获取该类的采样倍数（默认用 times）
            for i in range(int(mul // 1)):  # 整数倍部分，重复加入所有样本
                indices.extend(class_indices)
            rem = int((mul % 1) * len(class_indices))  #  # 余数部分，随机加入部分样本
            inds = torch.randperm(len(class_indices), generator=g).tolist()  # 打乱索引顺序
            indices.extend([class_indices[inds[i]] for i in range(rem)])  # 添加小数部分样本
       # 如果需要打乱整个索引列表
        if self.shuffle:
            shuffle = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle]

        # reset num_samples and total_size here.
        self.num_samples = math.ceil(len(indices) / self.num_replicas)
        # 计算所有进程总共需要的样本数量(可能大于实际样本数)
        self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible# 添加额外的样本，确保可以被进程数整除
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample   # 子采样：每个进程只取其负责的部分
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
