_base_ = [
    './kitti-3d-3class-lc.py',
]

voxel_size = [0.16, 0.16, 4]
point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]

model = dict(
    type='BEVF_FasterRCNN',
    camera_stream=False,                        # Turn off expensive LSS
    img_backbone=dict(
        type='TIMMBackbone',
        model_name='efficientnet_es',
        features_only=True,
        out_indices=(0, 1, 2),
    ),

    pts_voxel_layer=dict(                       # Same as pointpillar-lighter
        # max_num_points=32,
        max_num_points=28,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(                     # From BEVF
        type='HardVFE',
        in_channels=4,
        # feat_channels=[64, 64],
        feat_channels=[64],
        with_distance=False,
        # with_cluster_center=True,
        # with_voxel_center=True,
        with_cluster_center=True,
        with_voxel_center=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        fusion_layer=dict(
            type='PointFusion',
            img_channels=[24, 32, 48],
            pts_channels=64,
            mid_channels=32,
            out_channels=64,
            img_levels=[0, 1, 2],
            lateral_conv=False,
            randomized_fusion=False,
        )),
    # pts_voxel_encoder=dict(                     # From BEVF
    #     type='PillarFeatureNet',
    #     in_channels=4,
    #     feat_channels=[64],
    #     with_distance=False,
    #     voxel_size=voxel_size,
    #     point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
        # type='PointPillarsScatter', in_channels=64, output_shape=[304, 304]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 3],
        layer_strides=[2, 2],
        out_channels=[64, 128]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        # out_channels=[64, 64]),
        out_channels=[128, 128]),
        # out_channels=[64, 128]),

    # img_backbone=dict(
    #     type='TimmNet',
    #     model_name='lcnet_100',
    #     pretrained=True,
    # ),

    # img_backbone=dict(
    #     type='ResNet',
    #     depth=18,
    #     num_stages=2,
    #     strides=(1, 2),
    #     dilations=(1, 1),
    #     out_indices=(0, 1),
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    # ),

    # # img_neck=dict(
    # #     type='FPN',
    # #     in_channels=[256, 512, 1024, 2048],
    # #     # out_channels=256,
    # #     out_channels=imc,
    # #     num_outs=4,
    # # ),

    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        # in_channels=384,
        # feat_channels=384,
        in_channels=256,
        feat_channels=256,
        # in_channels=192,
        # feat_channels=192,
        # in_channels=128,
        # feat_channels=128,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
            sizes=[
                [0.8, 0.6, 1.73],   # Pedestrian
                [1.76, 0.6, 1.73],  # Cyclist
                [3.9, 1.6, 1.56],   # Car
            ],
            # custom_values=[0, 0],
            rotations=[0, 1.57],
            # reshape_out=True),
            reshape_out=False),
        # assigner_per_size=False,
        diff_rad_by_sin=True,
        # dir_offset=0.7854,  # pi/4
        # dir_limit_offset=0,
        # bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            # code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50))
)

data = dict(
    # samples_per_gpu=8,
    # workers_per_gpu=16,
    samples_per_gpu=24,
    workers_per_gpu=12,
    train=dict(times=10)
)

# lr = 0.001
lr = 0.0001
optimizer = dict(lr=lr)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))

# load_lift_from = 'checkpoints/bevf_pp_4x8_2x_nusc_cam_epoch_24.pth'     #####load cam stream
# load_from = 'checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'  #####load lidar stream

# runner = dict(type='EpochBasedRunner', max_epochs=25)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                name='pp-lc-monodepth-pgd___pretrain-both___lambda',
                dir='/data/disk/logs',
                project='mmDet3D',
                anonymous=None,
                reinit=True,
                id=None,
                notes='',
                resume='allow',
                tags=['KITTI', 'LiDAR+Camera'],
                entity='soonminh129',
                group='PointPillar-Lighter',
            )
        )
    ]
)

find_unused_parameters = True
