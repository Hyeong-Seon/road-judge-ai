custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmdet.models.losses',
    ])
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'))
default_scope = 'mmaction'
launcher = 'none'
load_from = '/content/drive/MyDrive/cv_final/work_dirs/model4_object_b_feature_FT2_Plus3/best_acc_top1_epoch_18.pth'
model = dict(
    backbone=dict(
        conv1_kernel=(
            5,
            7,
            7,
        ),
        conv1_stride_t=2,
        depth=50,
        inflate=(
            (
                1,
                1,
                1,
            ),
            (
                1,
                0,
                1,
                0,
            ),
            (
                1,
                0,
                1,
                0,
                1,
                0,
            ),
            (
                0,
                1,
                0,
            ),
        ),
        norm_eval=False,
        pool1_stride_t=2,
        pretrained='torchvision://resnet50',
        pretrained2d=True,
        type='ResNet3d'),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.7,
        in_channels=2048,
        init_std=0.01,
        loss_cls=dict(
            alpha=0.25,
            gamma=2.5,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        num_classes=79,
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=5e-06, type='AdamW', weight_decay=0.1))
param_scheduler = [
    dict(T_max=15, begin=0, by_epoch=True, end=15, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='/content/val_object_b_feature.txt',
        data_prefix=dict(video='/content/data_local/DATA(bb_1)_224_CPU/val'),
        pipeline=[
            dict(type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=2,
                num_clips=3,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    metric_list=(
        'top_k_accuracy',
        'mean_class_accuracy',
    ), type='AccMetric')
train_cfg = dict(
    max_epochs=15, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='/content/train_object_b_feature.txt',
        data_prefix=dict(video='/content/data_local/DATA(bb_1)_224_CPU/train'),
        pipeline=[
            dict(type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=2,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                type='ColorJitter'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='DecordInit'),
    dict(clip_len=16, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(brightness=0.3, contrast=0.3, saturation=0.3, type='ColorJitter'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='/content/val_object_b_feature.txt',
        data_prefix=dict(video='/content/data_local/DATA(bb_1)_224_CPU/val'),
        pipeline=[
            dict(type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=2,
                num_clips=3,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metric_list=(
        'top_k_accuracy',
        'mean_class_accuracy',
    ), type='AccMetric')
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=2,
        num_clips=3,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
work_dir = '/content/drive/MyDrive/cv_final/work_dirs/model4_object_b_feature_FT3_Extreme'
