# CMU 3D Object Detection Repo

## Usage
Demo example: 
Given image, point cloud, and calibration file, demo file conducts 3D object detection and saves the results.
```bash
python detection_3d_demo.py demo/data/kitti_000008.bin demo/data/kitti_000008.png demo/data/kitti_calibration.yaml configs/bevf_pp_lighter_effnet-es_kitti.py checkpoints/epoch_4.pth --snapshot
```

ROS example: 
Given RGB and LiDAR sensor topics, publish 3d detection result topic.
```bash
python detection_3d_ros.py --config configs/bevf_pp_lighter_effnet-es_kitti.py --calib demo/data/kitti/kitti_calibration.yaml --checkpoint checkpoints/epoch_4.pth 
```

Note: Some codes are dataset-specific, minor modification is required for the new dataset or hardware.
Note: Pre-defined 3D detection msg is not compatible with the prediction results of 3D detection network. Minor modification is required.

## Installtion
```bash
git clone <GIT_ADDRESS>
pip install -e .
```

## Dependencies
Device: Nvidia Jetson AGX Orin
- L4T: 35.1.0
- Jetpack: 5.0.2
- CUDA: 11.4.239
- Ubuntu: 20.04.6 LTS
- ROS: noetic 1.15.14
- Torch: 1.12.0a0+8a1a939.nv22.5
- TensorRT: 8.4.1.5
- Python: 3.8.10
- MMCV-full: 1.6.1
- MMdet: 2.25.2

Docker image source: https://hub.docker.com/r/dustynv/ros/tags
- Tag: noetic-pytorch-l4t-r35.1.0
Current docker image: TBA
