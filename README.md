# CMU 3D Object Detection & Tracking Repo

## Usage
Demo example: 

Given image, point cloud, and calibration file, the demo file conducts 3D object detection/tracking and saves the results.
```bash
# 1. 3D object detection demo, results are saved at demo/results folder.
python detection_3d_demo.py demo/data_det/kitti_000008.bin demo/data_det/kitti_000008.png demo/data_det/kitti_calibration.yaml configs/bevf_pp_lighter_effnet-es_kitti.py checkpoints/epoch_4.pth --snapshot
```

```bash
# 2. 3D object tracking demo
python tracker_3d_demo.py demo/data_trk/lidar demo/data_trk/img demo/data_trk/kitti_calibration.yaml configs/bevf_pp_lighter_effnet-es_kitti.py checkpoints/epoch_4.pth
```

ROS example: 

Given RGB and LiDAR sensor topics, publish 3D tracking result topic.
```bash
# 1. 3D object detection 
python detection_3d_ros.py --config configs/bevf_pp_lighter_effnet-es_kitti.py --calib demo/data/kitti/kitti_calibration.yaml --checkpoint checkpoints/epoch_4.pth 

# 2. 3D object tracking
python tracker_3d_ros.py --config configs/bevf_pp_lighter_effnet-es_kitti.py --calib demo/data/kitti/kitti_calibration.yaml --checkpoint checkpoints/epoch_4.pth 
```

Note: Some codes are dataset-specific. Minor modification is required for the new dataset or hardware.

Note: The pre-defined 3D detection msg is not compatible with the prediction results of 3D detection network. Minor modification is required.

## Installation
```bash
# 1. docker installation
cd ${WORK_DIR}
git clone https://github.com/dusty-nv/jetson-inference.git
docker pull dustynv/ros:noetic-pytorch-l4t-r35.1.0

# 2. tracking code installation
git clone https://github.com/UkcheolShin/CMU_3D_Det.git
cd jetson-inference/
docker/run.sh -c dustynv/ros:noetic-pytorch-l4t-r35.1.0 --ros=noetic --volume ${WORK_DIR}:/results
cd ${WORK_DIR}/CMU_3D_Det
pip install -e .
```

## Dependencies
Device: Nvidia Jetson AGX Orin
- L4T: 35.1.0
- Jetpack: 5.0.2
- CUDA: 11.4.239
- Ubuntu: 20.04.6 LTS
- ROS: noetic 1.15.14

Docker:
- Torch: 1.12.0a0+8a1a939.nv22.5
- TensorRT: 8.4.1.5
- Python: 3.8.10
- MMCV-full: 1.6.1
- MMdet: 2.25.2

Docker image source: https://hub.docker.com/r/dustynv/ros/tags
- Tag: noetic-pytorch-l4t-r35.1.0

