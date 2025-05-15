#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))    #随机生成一个在 12000 到 31999 之间的端口号，赋值给环境变量 MASTER_PORT，用于分布式训练中主节点的通信端口
export CUDA_VISIBLE_DEVICES=2   #指定gpu的编号
set -x  #启用bash的调试模式，会执行的每个命令打印出来，方便调试脚本

CONFIG=$1   #将第一个参数（表示配置文件的路径）赋值给 CONFIG
GPUS=$2   #将第二个参数（表示要使用的 GPU 数量）赋值给 GPUS

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
