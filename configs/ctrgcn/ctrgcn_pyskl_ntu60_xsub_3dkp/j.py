model = dict(
    type='RecognizerGCN',    #  # 识别器
    backbone=dict(
		type='CTRGCN',   # 使用CTRGCN作为骨干网络
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),   # 使用CTRGCN作为骨干网络
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))   # 分类头：GCN头，60个类别，输入通道为256

# 设置数据集类型
dataset_type = 'PoseDataset'  # 使用姿态数据集
ann_file = '/dadaY/xinyu/dataset/ntu60_pkl/ntu60_3danno.pkl'  # 数据集标注文件路径
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100),       # 均匀采样100帧
    dict(type='PoseDecode'),  # 姿态解码
    dict(type='FormatGCNInput', num_person=2),  # 格式化GCN输入，每个样本最多2人
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),  # 收集关键点和标签信息
    dict(type='ToTensor', keys=['keypoint'])   # 转换关键点数据为张量
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/j'
