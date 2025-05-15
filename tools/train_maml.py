import argparse
import os
import os.path as osp
import time
import torch

import mmcv
from mmcv import Config
from mmcv.runner import set_random_seed

from pyskl.datasets import build_dataset, build_dataloader
from pyskl.datasets.meta_dataset import MetaTaskDataset
from pyskl.utils import collect_env, get_root_logger
from maml.Meta import Meta

def parse_args():
    parser = argparse.ArgumentParser(description='Train MAML for skeleton-based action recognition')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # 设置随机种子
    if args.seed is not None:
        print(f'Set random seed to {args.seed}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # 初始化日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    # 记录配置信息
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 构建训练数据集
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val) if args.validate else None

    # 元学习参数
    meta_args = mmcv.ConfigDict({
        'update_lr': cfg.meta_learning.update_lr,
        'meta_lr': cfg.meta_learning.meta_lr,
        'n_way': cfg.meta_learning.n_way,
        'k_spt': cfg.meta_learning.k_shot,
        'k_qry': cfg.meta_learning.k_query,
        'task_num': cfg.meta_learning.task_num,
        'update_step': cfg.meta_learning.update_step,
        'update_step_test': cfg.meta_learning.update_step_test,
    })

    # 创建元学习任务数据集
    meta_train_dataset = MetaTaskDataset(
        base_dataset=train_dataset,
        n_way=cfg.meta_learning.n_way,
        k_shot=cfg.meta_learning.k_shot,
        k_query=cfg.meta_learning.k_query,
        task_num=cfg.meta_learning.task_num
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        meta_train_dataset,
        batch_size=1,  # 对于元学习，每个batch已经包含了多个任务
        shuffle=True,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True
    )
    
    if val_dataset is not None:
        meta_val_dataset = MetaTaskDataset(
            base_dataset=val_dataset,
            n_way=cfg.meta_learning.n_way,
            k_shot=cfg.meta_learning.k_shot,
            k_query=cfg.meta_learning.k_query,
            task_num=1  # 验证时每次只使用一个任务
        )
        val_loader = torch.utils.data.DataLoader(
            meta_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.workers_per_gpu,
            pin_memory=True
        )
    else:
        val_loader = None

    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 定义learner参数
    learner_args = {
        'input_features': 3,  # 假设keypoint是3D坐标
        'mlp_hidden': 64,
        'gcn_hidden': 128,
        'output_embedding_dim': 256,
        'num_joints': 25  # NTU RGB+D 有25个关节
    }
    
    # 创建MAML模型
    maml = Meta(meta_args, learner_args).to(device)
    
    # 训练循环
    for epoch in range(cfg.total_epochs):
        logger.info(f'Epoch {epoch+1}/{cfg.total_epochs}')
        
        # 训练一个epoch
        maml.train()
        train_losses = []
        
        for i, batch in enumerate(train_loader):
            # 将数据移至GPU
            x_spt = batch['x_spt'].squeeze(0).to(device)  # [task_num, n_way*k_shot, num_joints, feature_dim]
            y_spt = batch['y_spt'].squeeze(0).to(device)  # [task_num, n_way*k_shot]
            adj_spt = batch['adj_spt'].squeeze(0).to(device)  # [task_num, n_way*k_shot, num_joints, num_joints]
            x_qry = batch['x_qry'].squeeze(0).to(device)  # [task_num, n_way*k_query, num_joints, feature_dim]
            y_qry = batch['y_qry'].squeeze(0).to(device)  # [task_num, n_way*k_query]
            adj_qry = batch['adj_qry'].squeeze(0).to(device)  # [task_num, n_way*k_query, num_joints, num_joints]
            
            # 前向传播和训练
            accs = maml(x_spt, y_spt, adj_spt, x_qry, y_qry, adj_qry)
            
            if (i + 1) % 10 == 0:
                logger.info(f'Step: {i+1}, Meta-Train Accuracy: {accs[-1]:.4f}')
        
        # 验证
        if val_loader is not None and (epoch + 1) % cfg.evaluation.interval == 0:
            maml.eval()
            val_accs = []
            
            for batch in val_loader:
                x_spt = batch['x_spt'].squeeze(0).to(device)  # [1, n_way*k_shot, num_joints, feature_dim]
                y_spt = batch['y_spt'].squeeze(0).to(device)  # [1, n_way*k_shot]
                adj_spt = batch['adj_spt'].squeeze(0).to(device)  # [1, n_way*k_shot, num_joints, num_joints]
                x_qry = batch['x_qry'].squeeze(0).to(device)  # [1, n_way*k_query, num_joints, feature_dim]
                y_qry = batch['y_qry'].squeeze(0).to(device)  # [1, n_way*k_query]
                adj_qry = batch['adj_qry'].squeeze(0).to(device)  # [1, n_way*k_query, num_joints, num_joints]
                
                # 使用finetunning进行评估
                with torch.no_grad():
                    accs = maml.finetunning(x_spt[0], y_spt[0], adj_spt[0], x_qry[0], y_qry[0], adj_qry[0])
                val_accs.append(accs)
            
            val_acc = torch.tensor(val_accs).mean(dim=0)
            logger.info(f'Epoch {epoch+1}, Meta-Val Accuracy: {val_acc[-1]:.4f}')
        
        # 保存检查点
        if (epoch + 1) % cfg.checkpoint_config.interval == 0:
            checkpoint_path = osp.join(cfg.work_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'maml_state_dict': maml.state_dict(),
                'learner_state_dict': maml.net.state_dict()
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

if __name__ == '__main__':
    main()