#!/usr/bin/env python3
# Written by Ukcheol Shin (ushin@andrew.cmu.edu)

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from perception_msg.msg import bounding_box_3d, dynamic_object, dynamic_objects

import numpy as np
import cv2
from argparse import ArgumentParser
from mmdet3d.apis import inference_multi_modality_detector, init_model
from kf_tracker.tracker import AB3DMOT
from kf_tracker.utils import proj_lidar_bbox3d_on_img, save_image_with_boxes
from mmdet3d.core import LiDARInstance3DBoxes
import warnings
warnings.filterwarnings("ignore")

class ros_detection3d_node():
    def __init__(self, options):
        print("Trk3D: Initialization start for 3D objection tracking node..")
        self.args = options
        self.calb = options.calib
        self.score_thr = options.score_thr

        # 1. Initialize inference model
        # build the model from a config file and a checkpoint file
        self.det_model = init_model(self.args.config, self.args.checkpoint, 
                                device=self.args.device)
        rospy.loginfo('Trk3D: Pretrained 3D object detector loaded...')

        self.trk_model = AB3DMOT()
        rospy.loginfo('Trk3D: 3D object tracker loaded...')

        # 2. Initialize subscriber/publisher ROS nodes
        self.name_node = 'Trk3D_ver0'
        rospy.init_node(self.name_node, anonymous=False)

        # Maximum acceptable time difference in secs
        self.t_diff_max = rospy.get_param('~t_diff_max', 0.1)
        self.queue_size = rospy.get_param('~queue_size', 10)

        # Assume RGB topic name is '/cam1/image/compressed'/' with CompressedImage type message
        # Assume Lidar topic name is '/cam1/image/compressed'/' with PointCloud2 type message
        topic_sub_rgb   = rospy.get_param('~topic_rgb','/cam1/image/compressed')
        topic_sub_lidar = rospy.get_param('~topic_lidar','/lidar1/velodyne_points')
        topic_pub = rospy.get_param('~topic_det3d','/Trk3D/pred_results')

        # synchronized subscriber
        sub_rgb   = message_filters.Subscriber(topic_sub_rgb, CompressedImage,
                                               queue_size=self.queue_size)
        sub_lidar = message_filters.Subscriber(topic_sub_lidar, PointCloud2, 
                                               queue_size=self.queue_size)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_lidar], queue_size=self.queue_size, slop=self.t_diff_max)
        self.sync.registerCallback(self.callback)
        # self.pub = rospy.Publisher(topic_pub, dynamic_objects, queue_size=1)

        rospy.loginfo('Trk3D: Node information')
        rospy.loginfo('Trk3D: sub_RGB : {}'.format(topic_sub_rgb))
        rospy.loginfo('Trk3D: sub_LiDAR : {}'.format(topic_sub_lidar))
        rospy.loginfo('Trk3D: pub_pred : {}'.format(topic_pub))
        rospy.loginfo('Trk3D: t_diff_max : {}'.format(self.t_diff_max))
        rospy.loginfo('Trk3D: queue_size : {}'.format(self.queue_size))
        rospy.loginfo('Trk3D: {} node configuration done...'.format(self.name_node))

    def callback(self, msg_img: CompressedImage, msg_lidar: PointCloud2):
        try:
            # Convert images
            image = cv2.imdecode(np.fromstring(msg_img.data, np.uint8), cv2.IMREAD_UNCHANGED)

            # Get LiDARs
            lidar_pc = []
            for x, y, z, v, r, t in pc2.read_points(msg_lidar, field_names=('x', 'y', 'z', 
                                                    'intensity', 'ring', 'time'),
                                                    skip_nans=True):
                lidar_pc.append([float(x), float(y), float(z), float(v), float(r), float(t)])
            lidar_pc = np.array(lidar_pc)

            # inference 3d object detection
            det_result, meta_data = inference_multi_modality_detector(self.det_model, lidar_pc, image, self.calib)
            det_result_np = self.convert_dets_to_box_data(det_result, threshold=self.score_thr)

            # inference 3d object tracking
            start_time = time.time()
            trk_result = self.trk_model.update(det_result_np)
            cycle_time = time.time() - start_time

            # publish result
            objs = self.create_dynamic_objs_msg(trk_result, meta_data)
            self.pub.publish(objs)
        except:
            pass

    def convert_dets_to_box_data(self, det_result, threshold=0.4):
        """
        Converting dataformat for tracking module.
        """
        dets_all = {'dets': [], 'confs': [], 'labels': []}

        det_result = det_result[0]['pts_bbox']
        valid_objs = det_result['scores_3d'] > threshold

        dets_all['dets'] = det_result['boxes_3d'][valid_objs].tensor.numpy() 
        dets_all['confs'] = det_result['scores_3d'][valid_objs].numpy() 
        dets_all['labels'] = det_result['labels_3d'][valid_objs].numpy()

        return dets_all

    # Note: current 3d detection output is not matched to the pre-defined 3d bbox message file
    def create_bounding_3d_msg(self, box):
        out = bounding_box_3d()
        out.x = box[0]
        out.y = box[1]
        out.z = box[2]
        out.x_size = box[3]
        out.y_size = box[4]
        out.z_size = box[5]
        out.yaw = box[6]
        return out 

    # Note: current class ID is not matched to the pre-defined class ID (dynamic_object.msg file) 
    def create_dynamic_objs_msg(self, trk_result, meta_data):
        # post-process output data 
        """
        trk_result : Predicted results and data from pipeline (Nx10, N: tracked objects)
                   : containing following results: [x, y, z, x_size, y_size, z_size, yaw, id, score, label]
                 : (x, y, z, x_size, y_size, z_size, yaw) in Lidar coodinates 
        'id'     : ID for tracked objects
        'scores' : confidence score (Nx1), 0 is low, 1 is high confidence 
        'label'  : label of detected objects (Nx1), 0: pedestrian, 1: cylist, 2:car 
        bboxes_2d: (N,8,2) array of vertices for the 3d box in following order:
                    6 -------- 5 
                   /|         /|
                  2 -------- 1 .
                  | |        | |
                  . 7 -------- 4
                  |/         |/
                  3 -------- 0   
        """
        bboxes_3d  = LiDARInstance3DBoxes(trk_result[:,:7], origin=(0.5, 0.5, 0))
        bboxes_2d  = proj_lidar_bbox3d_on_img(bboxes_3d, meta_data['img_metas'][0][0]['lidar2img'])
        bboxes_cvx_hull = bboxes_3d.corners #(see details: mmdet3d/core/bbox/lidar_box3d.py)
        ids    = trk_result[:,7].astype(np.uint8)
        scores = trk_result[:,8]
        labels = trk_result[:,9]

        objs = dynamic_objects()
        objs.header.stamp = rospy.Time.now()
        for box_3d, box_2d, score, idx, label in zip(bboxes_3d,bboxes_2d,scores,ids,labels):
            obj = dynamic_object()
            obj.object_id = idx 
            obj.object_class = int(label)
            obj.object_score = float(score)
            obj.object_3d = self.create_bounding_3d_msg(box_3d)
            objs.object.append(obj)
        return objs

def get_options():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--calib', help='calibration file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0, help='bbox score threshold')
    return parser.parse_args()

def main():
    options = get_options()
    ros_detection3d_node(options)
    rospy.spin()

if __name__ =='__main__':
    main()