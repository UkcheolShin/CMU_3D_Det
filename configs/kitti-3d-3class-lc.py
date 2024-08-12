# dataset settings
class_names = ['Pedestrian', 'Cyclist', 'Car']

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

img_scale = (1280, 384)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', project_pts_to_img_depth=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img']),
            dict(type='Point2Voxel',
                 max_num_points=28,
                 point_cloud_range=point_cloud_range,
                 voxel_size=[0.16, 0.16, 4],
                 max_voxel=40000,
                 )
        ])
]

data = dict(
    test=dict(
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))