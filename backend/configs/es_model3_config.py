ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'my_modules',
    ])
data_root_val = 'data/kinetics400/videos_val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        out_dir='/content/training_temp_A',
        save_best='acc/mean1',
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
launcher = 'none'
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
        loss_cls=dict(
            cls_num_list=[
                129,
                915,
                1226,
                172,
                516,
                347,
                141,
                19,
                102,
                3095,
                276,
                634,
                21,
                2,
                205,
                340,
                146,
                41,
                206,
                202,
                34,
                14,
                21,
                37,
                499,
                50,
                19,
                16,
                26,
                9,
                6,
                11,
                11,
                1,
                3,
                5,
                8,
                29,
                48,
                243,
                9,
                18,
                30,
                73,
                37,
                41,
                20,
                9,
                223,
                543,
                73,
                19,
                66,
                27,
                72,
                120,
                13,
                268,
                153,
                40,
                121,
                27,
                13,
                10,
                35,
                81,
                76,
                79,
                40,
                143,
                4,
                12,
                44,
                60,
                35,
                61,
                22,
                40,
                8,
            ],
            loss_weight=1.0,
            max_m=0.5,
            s=30,
            type='LDAMLossCustom'),
        num_classes=79,
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
    accumulative_counts=2,
    clip_grad=dict(max_norm=40),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(T_max=25, by_epoch=True, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='/content/drive/MyDrive/cv_final/val.txtA',
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
train_cfg = dict(max_epochs=25, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='/content/drive/MyDrive/cv_final/train.txtA',
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
    batch_size=16,
    dataset=dict(
        ann_file='/content/drive/MyDrive/cv_final/val.txtA',
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
work_dir = '/content/training_temp_A'
