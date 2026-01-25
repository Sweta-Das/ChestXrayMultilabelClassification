# model settings (architecture)
model = dict(
    type='ImageClassifier', # overall model type

    # Feature extractor network
    backbone=dict(
        type='my_hybird_CTransCNN',
        arch='tiny',
        patch_size=16,  # patch_size=32 # size of image patches for transformer input
        drop_path_rate=0.1
    ),

    # Intermediate layer between backbone and head
    neck=None,

    # Classification head
    head=dict(
        type='My_Hybird_Head',
        num_classes=11,
        in_channels=[256, 384], # input channels from backbone
        init_cfg=None,
        loss=dict(
            type='BCE_ASL_Focal', reduction='mean', loss_weight=1),
        cal_acc=False),
    
    # Initialization for layers (truncated normal for linear, constant for LayerNorm)
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)

# dataset settings
dataset_type = 'My_MltilabelData'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# data pipelines

# Sequence of data transformations for training.
## load image, resize, randonly flip, normalize, convert to tensor, collect image and label
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

## similar to train, but no random flip, and only collects image
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

# data sources
data_root = './open_data/' # dataset fldr

# data loading
data = dict(
    samples_per_gpu=32, # batch-size per GPU
    # samples_per_gpu=16,
    workers_per_gpu=8, # data loader workers per GPU
    train=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=data_root + 'chest_x-rays11_kaggle_train.txt',
        classes=data_root + 'chest_x-rays11_multi_label_classes.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=data_root + 'chest_x-rays11_kaggle_val.txt',
        classes=data_root + 'chest_x-rays11_multi_label_classes.txt',
        pipeline=test_pipeline),
    test=dict(

        type=dataset_type,
        data_prefix=data_root,
        ann_file=data_root + 'chest_x-rays11_kaggle_test.txt',
        classes=data_root + 'chest_x-rays11_multi_label_classes.txt',
        pipeline=test_pipeline))

# Evaluation
## evaluate every epoch, use multiple metrics (mAP, class/overall precision/recall, F1, AUC)
evaluation = dict(interval=1, metric=[
    'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'multi_auc'
]) 

# Optimizer
## custom weight decay for normalization layers, biases, and class tokens
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
    })

## AdamW optimizer with specified hyperparams
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.5, 0.999),
    # betas=(0.5, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=None)

# learning rate policy
## cosine annealing learning rate, linear warmup for initial steps
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5 * 1252,
    warmup_by_epoch=False)

# checkpoint saving
runner = dict(type='EpochBasedRunner', max_epochs=300) # train for 300 epochs
checkpoint_config = dict(interval=1) # save model every epoch
# log every 100 iterations using text logger
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# Distributed training and Misc
dist_params = dict(backend='nccl') # use NCCL backend for distributed training
log_level = 'INFO' 
load_from = None # path to load pre-trained weights
resume_from = None # resume from checkpoint
workflow = [('train', 1)] # only training phase, run once per epoch
