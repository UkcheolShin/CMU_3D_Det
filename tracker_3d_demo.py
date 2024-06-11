# Written by Ukcheol Shin (ushin@andrew.cmu.edu)
import time, os
import numpy as np
from argparse import ArgumentParser
from mmdet3d.apis import (inference_multi_modality_detector, init_model,
                          show_result_meshlab)
from kf_tracker.tracker import AB3DMOT
from kf_tracker.utils import proj_lidar_bbox3d_on_img, save_image_with_boxes
from mmdet3d.core import LiDARInstance3DBoxes
import warnings
warnings.filterwarnings("ignore")

def convert_dets_to_box_data(det_result, threshold=0.4):
    dets_all = {'dets': [], 'confs': [], 'labels': []}

    det_result = det_result[0]['pts_bbox']
    valid_objs = det_result['scores_3d'] > threshold

    dets_all['dets'] = det_result['boxes_3d'][valid_objs].tensor.numpy() 
    dets_all['confs'] = det_result['scores_3d'][valid_objs].numpy() 
    dets_all['labels'] = det_result['labels_3d'][valid_objs].numpy()

    return dets_all

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo/results', help='dir to save results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device) # detector
    mot_tracker = AB3DMOT() # tracker
    print('Model initialization done...')

    flielist = os.listdir(args.image)
    flielist.sort()
    for filename in flielist:
        file_name  = os.path.split(filename)[-1].split('.')[0]
        path_img   = os.path.join(args.image, file_name+".png")
        path_lidar = os.path.join(args.pcd, file_name+".bin")
        path_calib = args.ann

        # raw sensor forwarding test
        # import mmcv
        # import numpy as np
        # path_img = mmcv.imread(path_img, 'unchanged')
        # path_lidar = np.fromfile(path_lidar, dtype=np.float32)

        # test a single image
        det_result, data = inference_multi_modality_detector(model, path_lidar,
                                                         path_img, path_calib)
        det_result_np = convert_dets_to_box_data(det_result, threshold=args.score_thr) # assume single batch size

        # important
        start_time = time.time()
        trk_result = mot_tracker.update(det_result_np)
        cycle_time = time.time() - start_time

        bboxes_3d  = LiDARInstance3DBoxes(trk_result[:,:7], origin=(0.5, 0.5, 0))
        bboxes_vertex = bboxes_3d.corners
        bboxes_2d  = proj_lidar_bbox3d_on_img(bboxes_3d, data['img_metas'][0][0]['lidar2img'])
        obj_ids    = trk_result[:,7].astype(np.uint8)
        obj_scores = trk_result[:,8]
        obj_labels = trk_result[:,9]

        save_image_with_boxes(path_img,
                              bboxes_2d,
                              obj_ids, 
                              obj_scores, 
                              result_path=args.out_dir)

        trk_result = mot_tracker.update(det_result_np)

    print('Prediction result is saved...')

if __name__ == '__main__':
    main()
