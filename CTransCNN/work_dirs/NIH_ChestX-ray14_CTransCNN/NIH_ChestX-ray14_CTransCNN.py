model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='my_hybird_CTransCNN',
        arch='tiny',
        patch_size=16,
        drop_path_rate=0.1),
    neck=None,
    head=dict(
        type='My_Hybird_Head',
        num_classes=14,
        in_channels=[256, 384],
        init_cfg=None,
        loss=dict(type='BCE_ASL_Focal', reduction='mean', loss_weight=1),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
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
data_root = './dataset/'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type='My_MltilabelData',
        data_prefix='./dataset/',
        ann_file='./dataset/chest14_train_labels.txt',
        classes='./dataset/chest14_classes.txt',
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
        data_prefix='./dataset/',
        ann_file='./dataset/chest14_val_labels.txt',
        classes='./dataset/chest14_classes.txt',
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
        data_prefix='./dataset/',
        ann_file='./dataset/chest14_test_labels.txt',
        classes='./dataset/chest14_classes.txt',
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
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys=dict({'.cls_token': dict(decay_mult=0.0)}))
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.5, 0.999),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({'.cls_token': dict(decay_mult=0.0)})))
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
resume_from = '/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN/work_dirs/NIH_ChestX-ray14_CTransCNN/epoch_284.pth'
workflow = [('train', 1)]
work_dir = '/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN/work_dirs/NIH_ChestX-ray14_CTransCNN'
gpu_ids = [0]
