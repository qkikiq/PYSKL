# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from pyskl import __version__
from pyskl.apis import init_random_seed
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port


def parse_args():
    # 创建一个参数解析器，用于解析命令行参数
    parser = argparse.ArgumentParser(description='Train a recognizer with few-shot learning')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')

    # 小样本学习参数组
    parser.add_argument(
        '--n-way',
        type=int,
        default=5,
        help='N-way in few-shot learning')
    parser.add_argument(
        '--k-shot',
        type=int,
        default=1,
        help='K-shot in few-shot learning')
    parser.add_argument(
        '--query-num',
        type=int,
        default=15,
        help='number of query samples per class')
    parser.add_argument(
        '--meta-test-iter',
        type=int,
        default=600,
        help='number of iterations during meta-testing')
    parser.add_argument(
        '--meta-val-iter',
        type=int,
        default=600,
        help='number of iterations during meta-validation')
    parser.add_argument(
        '--meta-train-iter',
        type=int,
        default=60000,
        help='number of iterations during meta-training')
    parser.add_argument(
        '--meta-algorithm',
        type=str,
        default='MAML',
        choices=['MAML', 'ProtoNet', 'RelationNet'],
        help='meta-learning algorithm')

    # 通用训练参数
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help='whether to test the best checkpoint after training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # 设置小样本学习配置
    cfg.few_shot = dict(
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_num=args.query_num,
        meta_algorithm=args.meta_algorithm,
        meta_test_iter=args.meta_test_iter,
        meta_val_iter=args.meta_val_iter,
        meta_train_iter=args.meta_train_iter
    )

    # 设置cudnn_benchmark以提高卷积效率
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 设置工作目录
    if cfg.get('work_dir', None) is None:
        # 如果未指定工作目录，使用配置文件名作为默认值
        cfg.work_dir = osp.join('./work_dirs/few_shot',
                                f"{osp.splitext(osp.basename(args.config))[0]}_{args.n_way}way_{args.k_shot}shot")

    # 设置分布式训练参数
    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')
    init_dist(args.launcher, **cfg.dist_params)

    # 获取分布式训练信息
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # 自动恢复训练
    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    # 创建工作目录并保存配置
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # 初始化日志
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'few_shot_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))

    # 记录元信息
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # 记录基本配置信息
    logger.info(f'Config: {cfg.pretty_text}')
    logger.info(f'Few-shot config: {args.n_way}-way {args.k_shot}-shot with {args.meta_algorithm}')

    # 设置随机种子
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    # 保存种子和配置信息
    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))
    meta['few_shot_config'] = cfg.few_shot

    # 构建模型
    logger.info(f'Building model for {args.meta_algorithm} meta-learning algorithm')
    model = build_model(cfg.model)

    # 编译模型（PyTorch 2.0+特性）
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        logger.info('Compiling model with PyTorch 2.0+')
        model = torch.compile(model)

    # 从pyskl.apis导入小样本学习相关函数
    from pyskl.apis import build_few_shot_dataset, train_few_shot_model, meta_test_model

    # 构建小样本学习数据集
    logger.info('Building few-shot datasets')

    # 判断当前是训练阶段还是测试阶段
    is_train = 'train' in cfg.workflow[0]

    if is_train:
        # 训练阶段构建元训练数据集和元验证数据集
        train_dataset = build_few_shot_dataset(
            cfg.data.train,
            mode='meta_train',
            n_way=args.n_way,
            k_shot=args.k_shot,
            query_num=args.query_num
        )

        val_dataset = None
        if args.validate and hasattr(cfg.data, 'val'):
            val_dataset = build_few_shot_dataset(
                cfg.data.val,
                mode='meta_val',
                n_way=args.n_way,
                k_shot=args.k_shot,
                query_num=args.query_num
            )

        datasets = [train_dataset, val_dataset] if val_dataset else [train_dataset]
    else:
        # 测试阶段构建元测试数据集
        test_dataset = build_few_shot_dataset(
            cfg.data.test,
            mode='meta_test',
            n_way=args.n_way,
            k_shot=args.k_shot,
            query_num=args.query_num
        )
        datasets = [test_dataset]

    # 检查点配置
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            pyskl_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
            few_shot_config=cfg.few_shot
        )

    # 测试选项
    test_option = dict(test_last=args.test_last, test_best=args.test_best)

    # memcached配置
    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    # 如果启用memcached，在主进程中启动memcached服务
    if rank == 0 and memcached:
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)

        # 等待memcached启动
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached.'

    # 同步所有进程
    dist.barrier()

    # 根据当前阶段执行对应操作
    if is_train:
        # 训练阶段
        logger.info(f'Starting meta-training with {args.meta_algorithm}')
        train_few_shot_model(
            model=model,
            datasets=datasets,
            cfg=cfg,
            validate=args.validate,
            timestamp=timestamp,
            meta=meta
        )
    else:
        # 测试阶段
        logger.info(f'Starting meta-testing with {args.meta_algorithm}')
        meta_test_model(
            model=model,
            dataset=datasets[0],
            cfg=cfg,
            logger=logger
        )

    # 同步所有进程
    dist.barrier()

    # 关闭memcached服务
    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()