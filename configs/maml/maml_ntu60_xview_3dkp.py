# 基础模型配置
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

# 数据集配置
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

# 数据处理流程
pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2)
]

# 元学习配置
meta_learning = dict(
    n_way=5,
    k_shot=5,
    k_query=15,
    task_num=4,
    update_step=5,
    update_step_test=10,
    meta_lr=0.001,
    update_lr=0.01
)

# 数据加载器配置
data = dict(
    workers_per_gpu=4,
    train=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        split='xview_train', 
        pipeline=pipeline),
    val=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        split='xview_val', 
        pipeline=pipeline),
    test=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        split='xview_val', 
        pipeline=pipeline)
)

# 训练配置
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/maml/maml_ntu60_xview_3dkp'