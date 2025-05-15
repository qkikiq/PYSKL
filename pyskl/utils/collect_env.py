# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash

import pyskl


def collect_env():
    # 调用MMCV库的collect_basic_env函数收集基本环境信息(如Python版本、CUDA版本等)
    env_info = collect_basic_env()
    env_info['pyskl'] = (
        # 格式为: 版本号+git哈希值(取前7位)
        pyskl.__version__ + '+' + get_git_hash(digits=7))
    # 返回包含所有环境信息的字典
    return env_info


if __name__ == '__main__':
    # 当直接运行此脚本时执行以下代码

    # 调用collect_env()函数获取环境信息
    # 遍历环境信息字典的所有键值对
    for name, val in collect_env().items():
        # 以"名称: 值"的格式打印每一项环境信息
        print(f'{name}: {val}')
