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

class ros_detection3d_node():
    def __init__(self, options):
        print("Det3D: Initialization start for 3D objection detection node..")
        self.args = options
        self.calb = options.calib
        self.score_thr = options.score_thr

        # 1. Initialize inference model
        # build the model from a config file and a checkpoint file
        self.model = init_model(self.args.config, self.args.checkpoint, 
                                device=self.args.device)
        rospy.loginfo('Det3D: Pretrained 3D object detector loaded...')

        # 2. Initialize subscriber/publisher ROS nodes
        self.name_node = 'Det3D_ver0'
        rospy.init_node(self.name_node, anonymous=False)

        # Maximum acceptable time difference in secs
        self.t_diff_max = rospy.get_param('~t_diff_max', 0.1)
        self.queue_size = rospy.get_param('~queue_size', 10)

        # Assume RGB topic name is '/cam1/image/compressed'/' with CompressedImage type message
        # Assume Lidar topic name is '/cam1/image/compressed'/' with PointCloud2 type message
        topic_sub_rgb   = rospy.get_param('~topic_rgb','/cam1/image/compressed')
        topic_sub_lidar = rospy.get_param('~topic_lidar','/lidar1/velodyne_points')
        topic_pub = rospy.get_param('~topic_det3d','/Det3D/pred_results')

        # synchronized subscriber
        sub_rgb   = message_filters.Subscriber(topic_sub_rgb, CompressedImage,
                                               queue_size=self.queue_size)
        sub_lidar = message_filters.Subscriber(topic_sub_lidar, PointCloud2, 
                                               queue_size=self.queue_size)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_lidar], queue_size=self.queue_size, slop=self.t_diff_max)
        self.sync.registerCallback(self.callback)
        # self.pub = rospy.Publisher(topic_pub, dynamic_objects, queue_size=1)

        rospy.loginfo('Det3D: Node information')
        rospy.loginfo('Det3D: sub_RGB : {}'.format(topic_sub_rgb))
        rospy.loginfo('Det3D: sub_LiDAR : {}'.format(topic_sub_lidar))
        rospy.loginfo('Det3D: pub_pred : {}'.format(topic_pub))
        rospy.loginfo('Det3D: t_diff_max : {}'.format(self.t_diff_max))
        rospy.loginfo('Det3D: queue_size : {}'.format(self.queue_size))
        rospy.loginfo('Det3D: {} node configuration done...'.format(self.name_node))

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

            # test a single image
            pred_result = inference_multi_modality_detector(self.model, lidar_pc, image, self.calib)[0]
            pred_result = pred_result[0]['pts_bbox'] # assume single batch size
            
            # pred_result contain following results : ['boxes_3d', 'scores_3d', 'labels_3d']
            # 'boxes_3d'  : Each row is (x, y, z, x_size, y_size, z_size, yaw) in Lidar coodinates (see details: mmdet3d/core/bbox/lidar_box3d.py)
            # 'scores_3d' : confidence score, 0 is low, 1 is high confidence 
            # 'label_id'  : 0: pedestrian, 1: cylist, 2:car 
            pred_bboxes = pred_result['boxes_3d'].tensor.numpy()
            pred_scores = pred_result['scores_3d'].numpy()
            label_ids   = pred_result['label_id'].numpy()

            # filter out low score bboxes if you need here
            if self.score_thr >= 0:
                inds = pred_scores > self.score_thr
                pred_bboxes = pred_bboxes[inds]
                pred_scores = pred_scores[inds]
                label_ids = label_ids[inds]

            objs = create_dynamic_objs_msg(pred_bboxes, pred_scores, label_ids)
            self.pub.publish(objs)
        except:
            pass

# Note: current 3d detection output is not matched to the pre-defined 3d bbox message file
def create_bounding_3d_msg(box):
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
def create_dynamic_objs_msg(boxes, scores, classes):
    objs = dynamic_objects()
    objs.header.stamp = rospy.Time.now()
    for idx, (box, score, cl) in enumerate(zip(boxes,scores,classes)):
        obj = dynamic_object()
        obj.object_id = idx 
        obj.object_class = int(cl)
        obj.object_score = float(score)
        obj.object_3d = create_bounding_3d_msg(box)
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