ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
data_root_val = 'data/kinetics400/videos_val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        out_dir='/content/training_temp',
        save_best=[
            'acc/top1',
            'acc/mean1',
        ],
        save_last=False,
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = '/content/drive/MyDrive/cv_finala/vit_b_fixed_v10.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        arch='base',
        depth=12,
        embed_dims=768,
        img_size=224,
        interpolate_pos_encoding=True,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        num_frames=16,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        type='VisionTransformer',
        with_cp=True),
    cls_head=dict(
        average_clips='prob',
        in_channels=768,
        num_classes=8,
        type='TimeSformerHead'),
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
    accumulative_counts=4,
    clip_grad=dict(max_norm=40),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(T_max=20, begin=0, by_epoch=True, end=20, type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/content/drive/MyDrive/cv_final/val.txtP',
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
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/content/drive/MyDrive/cv_final/train.txtP',
        data_prefix=dict(video=''),
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
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/content/drive/MyDrive/cv_final/val.txtP',
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
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metric_list=(
        'top_k_accuracy',
        'mean_class_accuracy',
    ), type='AccMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/content/training_temp'
