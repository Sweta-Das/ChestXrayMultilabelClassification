# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='My_Transformer_Decoder',
        arch='deit-tiny',
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=-1,
        with_cls_token=True,
        output_cls_token=True,
    ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=14,
        in_channels=192,
        loss=dict(
            type='BCE_ASL_Focal', reduction='mean', loss_weight=1),
        init_cfg=dict(type='Normal', layer='Linear', std=0.01),
    ),
)

# dataset settings
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
    dict(type='Normalize', **img_norm_cfg),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root = './dataset/'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=data_root + 'chest14_train_labels.txt',
        classes=data_root + 'chest14_classes.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=data_root + 'chest14_val_labels.txt',
        classes=data_root + 'chest14_classes.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=data_root + 'chest14_test_labels.txt',
        classes=data_root + 'chest14_classes.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=[
    'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'multi_auc'
])
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0),
    })
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.5, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5 * 1252,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=300)
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# Early stopping: stop if `multi_auc` doesn't improve for `patience` epochs.
# Set hook priority low so it runs after EvalHook.
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
