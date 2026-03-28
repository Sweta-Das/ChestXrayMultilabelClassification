model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNetV1d', depth=50, out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=14,
        in_channels=2048,
        init_cfg=None,
        loss=dict(type='BCE_ASL_Focal', reduction='mean', loss_weight=1),
        cal_acc=False))
dataset_type = 'My_MltilabelData'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(224, 224),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(224, 224),
        backend='pillow',
        interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data_root = '/var/tmp/pbs.803117.pbshpc/dataset/'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type='My_MltilabelData',
        data_prefix='/var/tmp/pbs.803117.pbshpc/dataset/',
        ann_file='/var/tmp/pbs.803117.pbshpc/dataset/chest14_train_labels.txt',
        classes='/var/tmp/pbs.803117.pbshpc/dataset/chest14_classes.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(224, 224),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='My_MltilabelData',
        data_prefix='/var/tmp/pbs.803117.pbshpc/dataset/',
        ann_file='/var/tmp/pbs.803117.pbshpc/dataset/chest14_val_labels.txt',
        classes='/var/tmp/pbs.803117.pbshpc/dataset/chest14_classes.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(224, 224),
                backend='pillow',
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='My_MltilabelData',
        data_prefix='/var/tmp/pbs.803117.pbshpc/dataset/',
        ann_file='/var/tmp/pbs.803117.pbshpc/dataset/chest14_test_labels.txt',
        classes='/var/tmp/pbs.803117.pbshpc/dataset/chest14_classes.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(224, 224),
                backend='pillow',
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(
    interval=1,
    metric=['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'multi_auc'])
optimizer = dict(
    type='AdamW', lr=0.001, weight_decay=0.05, eps=1e-08, betas=(0.5, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=6260,
    warmup_by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='multi_auc',
        rule='greater',
        patience=20,
        min_delta=0.0,
        start_epoch=1,
        interval=1,
        priority='LOWEST')
]
work_dir = './work_dirs/NIH_ChestX-ray14_ResNet50'
gpu_ids = [0]
