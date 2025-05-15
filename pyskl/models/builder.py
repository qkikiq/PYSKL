# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
        #PYSKL 框架中的模型注册表系统
MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
HEADS = MODELS
RECOGNIZERS = MODELS
LOSSES = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_recognizer(cfg):
    """Build recognizer."""
    return RECOGNIZERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_model(cfg):
    """Build model."""
    args = cfg.copy()  # 复制配置字典，避免修改原始配置
    obj_type = args.pop('type') # 从配置中弹出'type'键，获取模型类型名称
    if obj_type in RECOGNIZERS:    # 检查模型类型是否在已注册的识别器中
        return build_recognizer(cfg)
    raise ValueError(f'{obj_type} is not registered')
