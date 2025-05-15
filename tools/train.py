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
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port


def parse_args():
    # 创建一个参数解析器，用于解析命令行参数
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path') #配置文件的路径，用于指定训练所用的配置文件
    parser.add_argument(
        '--validate',  #训练过程是否验证
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',  #训练完成对最后一个检测点测试
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',  #训练完成后是否测试最佳模型
        action='store_true',
        help='whether to test the best checkpoint (if applicable) after training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')  #随机种子
    parser.add_argument(
        '--deterministic',  #是否设置确定性选项 保证结果可重复
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',   #启动方式的选择：用于分布式训练的任务启动器
        choices=['pytorch', 'slurm'],
        default='pytorch', #默认
        help='job launcher')
    parser.add_argument(
        '--compile',  #是否在训练/测试之前编译模型（仅适用于 pytorch 2.0）
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()

    # 如果环境变量中没有 LOCAL_RANK，则用传入的参数设置它
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # 解析命令行参数
    args = parse_args()

    # 从配置文件中加载配置
    cfg = Config.fromfile(args.config)

    # 如果配置中启用了 cudnn_benchmark，则设置为 True，提高卷积效率
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 设置工作目录（优先使用配置文件中的 work_dir，否则使用 config 文件名）
    # work_dir is determined in this priority:
    # config file > default (base filename)
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 如果没有分布式参数 dist_params，则设置默认 backend 为 nccl
    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    # 初始化分布式训练环境（如 torch.distributed）
    init_dist(args.launcher, **cfg.dist_params)

    # 获取当前进程的 rank 和总进程数（world size）
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)    #设置可用的gpu的id范围

    auto_resume = cfg.get('auto_resume', True)
    # 如果启用了 auto_resume，并且没有设置 resume_from，则尝试从 latest.pth 自动恢复
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config   将 config 文件复制一份保存到工作目录中（dump转存）
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    # 初始化日志文件，确保训练日志写入文件中
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())   #timestamp：记录当前时间
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')       #get_root_logger()：创建或获取一个日志记录器对象，写入指定文件。
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged

    # 初始化元信息，用于记录环境信息、种子等
    meta = dict()
    # log env info
    # 收集并记录环境信息（如 PyTorch、CUDA、Python 版本等）
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    # 打印配置文件内容
    logger.info(f'Config: {cfg.pretty_text}')  #cfg.pretty_text：美化后的配置文件内容字符串

    # set random seeds  设置随机种子，并记录到日志中
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    # 保存种子和配置信息到 meta
    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))


    # 根据配置文件中的 cfg.model 构建模型结构
    model = build_model(cfg.model)

    # 如果使用的是 PyTorch >= 2.0，并且设置了 --compile，则编译模型以提升性能
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    # 根据配置文件中的 cfg.data.train 构建训练数据集
    datasets = [build_dataset(cfg.data.train)]

    # 检查是否使用元学习
    use_maml = cfg.get('use_maml', False)




    # 设置 workflow，默认是 [('train', 1)] 表示训练 1 个 epoch
    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1

    # 如果配置文件中有 checkpoint_config，则设置保存检查点的元数据
    if cfg.checkpoint_config is not None:
        # save pyskl version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            #__version__：代码版本号
            pyskl_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)
    # 设置测试选项（是否在最后或最佳模型上测试）
    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    # memcached 默认配置（用于加速 pickle 数据访问）
    default_mc_cfg = ('localhost', 22077)
    # memcached 是一种分布式缓存服务，可加速数据读取
    memcached = cfg.get('memcached', False)

    # 如果当前是主进程（rank 0）并且启用了 memcached，则尝试启动它
    if rank == 0 and memcached:
        # mc_list is a list of pickle files you want to cache in memory.
        # Basically, each pickle file is a dictionary.
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)   #default_mc_cfg 是默认配置
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        # 检查 memcached 是否已经启动
        if not test_port(mc_cfg[0], mc_cfg[1]): #test_port()：测试指定端口是否可以连接
            mc_on(port=mc_cfg[1], launcher=args.launcher)  #mc_on()：启动 memcached 服务

        # 等待 memcached 启动，最多重试 3 次，每次等待 5 秒
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    # 所有进程同步，确保 memcached 和初始化已经完成
    dist.barrier()

    # 启动模型训练（包括验证与测试）
    #模型 数据集 配置 是否验证 测试选项 时间戳 元信息
    train_model(model, datasets, cfg, validate=args.validate, test=test_option, timestamp=timestamp, meta=meta)
    dist.barrier()
    # 如果主进程启用了 memcached，则关闭它
    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()
