import numpy as np
import torch
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose

@DATASETS.register_module()
class MetaTaskDataset(Dataset):
    """元学习任务数据集，用于生成支持集和查询集任务。"""
    
    def __init__(self,
                 base_dataset,
                 n_way,
                 k_shot,
                 k_query,
                 task_num,
                 pipeline=None):
        """
        初始化MetaTaskDataset。
        
        Args:
            base_dataset (Dataset): 基础数据集实例
            n_way (int): 任务中类别数量
            k_shot (int): 每个类别在支持集中的样本数
            k_query (int): 每个类别在查询集中的样本数
            task_num (int): 每次采样生成的任务数量
            pipeline (list[dict | callable]): 可选的数据处理流程
        """
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.task_num = task_num
        
        # 组织基础数据集中的样本，按类别分组
        self.cls_samples = {}
        for idx in range(len(base_dataset)):
            sample = base_dataset[idx]
            label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            if label not in self.cls_samples:
                self.cls_samples[label] = []
            self.cls_samples[label].append(idx)
        
        # 确保我们有足够的类别
        available_classes = list(self.cls_samples.keys())
        assert len(available_classes) >= n_way, f"数据集中类别数量{len(available_classes)}少于请求的n_way={n_way}"
        
        # 确保每个类别有足够的样本
        for cls, samples in self.cls_samples.items():
            assert len(samples) >= k_shot + k_query, f"类别{cls}的样本数量{len(samples)}少于k_shot+k_query={k_shot+k_query}"
        
        # 设置数据处理流程
        self.pipeline = Compose(pipeline) if pipeline else None
    
    def __len__(self):
        """返回数据集长度，在元学习中通常设置为一个较大的值，因为任务可以无限生成"""
        return 10000  # 设置一个较大的值，因为我们可以无限生成任务
    
    def __getitem__(self, idx):
        """
        生成一个元学习任务batch，包含多个任务，每个任务有支持集和查询集
        
        返回一个包含以下内容的字典:
            - x_spt: 支持集特征 [task_num, n_way*k_shot, ...]
            - y_spt: 支持集标签 [task_num, n_way*k_shot]
            - adj_spt: 支持集邻接矩阵 [task_num, n_way*k_shot, num_nodes, num_nodes]
            - x_qry: 查询集特征 [task_num, n_way*k_query, ...]
            - y_qry: 查询集标签 [task_num, n_way*k_query]
            - adj_qry: 查询集邻接矩阵 [task_num, n_way*k_query, num_nodes, num_nodes]
        """
        # 生成task_num个任务
        tasks = []
        for _ in range(self.task_num):
            # 随机选择n_way个类别
            available_classes = list(self.cls_samples.keys())
            selected_classes = np.random.choice(available_classes, self.n_way, replace=False)
            
            # 为每个任务创建支持集和查询集
            spt_samples = []
            qry_samples = []
            spt_labels = []
            qry_labels = []
            
            # 为每个选定的类别采样样本
            for i, cls in enumerate(selected_classes):
                # 获取该类别的所有样本索引
                cls_indices = self.cls_samples[cls]
                # 随机选择k_shot+k_query个样本
                selected_indices = np.random.choice(cls_indices, self.k_shot + self.k_query, replace=False)
                # 分割为支持集和查询集
                spt_indices = selected_indices[:self.k_shot]
                qry_indices = selected_indices[self.k_shot:]
                
                # 获取样本并添加到列表
                for idx in spt_indices:
                    sample = self.base_dataset[idx]
                    spt_samples.append(sample)
                    spt_labels.append(i)  # 使用相对标签(0到n_way-1)
                
                for idx in qry_indices:
                    sample = self.base_dataset[idx]
                    qry_samples.append(sample)
                    qry_labels.append(i)  # 使用相对标签
            
            # 处理支持集和查询集数据
            x_spt, adj_spt = self._process_samples(spt_samples)
            x_qry, adj_qry = self._process_samples(qry_samples)
            
            tasks.append({
                'x_spt': x_spt,
                'y_spt': torch.tensor(spt_labels),
                'adj_spt': adj_spt,
                'x_qry': x_qry,
                'y_qry': torch.tensor(qry_labels),
                'adj_qry': adj_qry
            })
        
        # 将所有任务组合成一个batch
        batch = {
            'x_spt': torch.stack([t['x_spt'] for t in tasks]),
            'y_spt': torch.stack([t['y_spt'] for t in tasks]),
            'adj_spt': torch.stack([t['adj_spt'] for t in tasks]),
            'x_qry': torch.stack([t['x_qry'] for t in tasks]),
            'y_qry': torch.stack([t['y_qry'] for t in tasks]),
            'adj_qry': torch.stack([t['adj_qry'] for t in tasks])
        }
        
        return batch
    
    def _process_samples(self, samples):
        """
        处理样本列表，提取特征和构建邻接矩阵
        
        Args:
            samples: 样本列表
            
        Returns:
            tuple: (特征张量, 邻接矩阵张量)
        """
        keypoints = []
        adjacency_matrices = []
        
        for sample in samples:
            # 提取关键点特征
            # 假设keypoint格式为[num_frames, num_persons, num_joints, feature_dim]
            keypoint = sample['keypoint']
            
            # 简化：取中间帧的第一个人
            if len(keypoint.shape) == 4:  # [num_frames, num_persons, num_joints, feature_dim]
                mid_frame = keypoint.shape[0] // 2
                keypoint = keypoint[mid_frame, 0]  # [num_joints, feature_dim]
            elif len(keypoint.shape) == 3:  # [num_frames, num_joints, feature_dim]
                mid_frame = keypoint.shape[0] // 2
                keypoint = keypoint[mid_frame]  # [num_joints, feature_dim]
            
            # 构建邻接矩阵 (这里需要根据您的具体骨架模型来构建)
            num_joints = keypoint.shape[0]
            adj_matrix = self._build_adjacency_matrix(num_joints, sample)
            
            keypoints.append(torch.tensor(keypoint, dtype=torch.float32))
            adjacency_matrices.append(torch.tensor(adj_matrix, dtype=torch.float32))
        
        # 堆叠成批次
        keypoints_tensor = torch.stack(keypoints)  # [batch_size, num_joints, feature_dim]
        adj_tensor = torch.stack(adjacency_matrices)  # [batch_size, num_joints, num_joints]
        
        return keypoints_tensor, adj_tensor
    
    def _build_adjacency_matrix(self, num_joints, sample):
        """
        根据骨架模型构建邻接矩阵
        
        Args:
            num_joints: 关节点数量
            sample: 样本数据，可能包含骨架连接信息
            
        Returns:
            numpy.ndarray: 邻接矩阵
        """
        # 这里实现根据您的骨架模型构建邻接矩阵的逻辑
        # 示例：使用预定义的骨架连接构建邻接矩阵
        
        # 为了简化，这里使用一个假设的骨架连接方式
        # 您需要根据实际的骨架模型替换这部分逻辑
        
        # 假设 sample['layout'] 包含骨架布局信息，例如 'nturgb+d' 或 'coco'
        layout = sample.get('layout', 'nturgb+d')  # 默认使用 NTU RGB+D 布局
        
        if layout == 'nturgb+d':
            # NTU RGB+D 的骨架连接 (简化版本)
            connections = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
                (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
                (24, 25), (25, 12)
            ]
            # 将索引调整为从0开始
            connections = [(a-1, b-1) for a, b in connections]
        elif layout == 'coco':
            # COCO 的骨架连接 (简化版本)
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6), (5, 11),
                (6, 12), (5, 6), (11, 12)
            ]
        else:
            # 如果未知布局，使用全连接图
            connections = [(i, j) for i in range(num_joints) for j in range(num_joints) if i != j]
        
        # 创建邻接矩阵
        adj_matrix = np.zeros((num_joints, num_joints), dtype=np.float32)
        for a, b in connections:
            if a < num_joints and b < num_joints:  # 确保索引在有效范围内
                adj_matrix[a, b] = 1
                adj_matrix[b, a] = 1  # 无向图
        
        # 添加自环
        for i in range(num_joints):
            adj_matrix[i, i] = 1
            
        return adj_matrix